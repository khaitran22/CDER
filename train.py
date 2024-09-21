import os
import argparse
import pickle
from tqdm import tqdm

from prepro import *
from utils import collate_fn, set_all_seeds
from evaluation import evaluate
from model import CDER

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoModel, AutoTokenizer
from transformers.optimization import get_linear_schedule_with_warmup

import wandb

import autoRun
autoRun.choose_gpu(retry=True, min_gpu_memory=10000, sleep_time=30)

def train(args, model, train_features, dev_features, test_features=None):
    best_evi_f1 = -1
    num_steps = 0
    new_layer = ['GCN', 'GAT', 'sentence_predictor', 'entity_pair_extractor']
    optimized_grouped_parameters = [
        {"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in new_layer)], },
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in new_layer)], "lr": args.grouped_learning_rate},
    ]
    optimizer = AdamW(optimized_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)

    train_dataloader = DataLoader(train_features, batch_size=args.train_batch_size, shuffle=True, collate_fn=collate_fn, drop_last=False)
    train_iterator = range(int(args.num_train_epochs))
    total_steps = int(len(train_dataloader) * args.num_train_epochs // args.gradient_accumulation_steps)
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)
    print("Total steps: {}".format(total_steps))
    print("Warmup steps: {}".format(warmup_steps))

    losses = []

    for epoch in train_iterator:
        model.train()
        for step, batch in enumerate(tqdm(train_dataloader, desc=f"Epoch {epoch+1}")):
            inputs = {
                'input_ids': batch[0].to(args.device),
                'attention_mask': batch[1].to(args.device),
                'labels': batch[2],
                'entity_pos': batch[3],
                'hts': batch[4],
                'sent_map': batch[5],
                'raw_evidence': batch[6],
                'gss_subgraph': batch[7],
                'pair_evidence': batch[8],
                'triple_evidence': batch[9],
                'gps_subgraph': batch[10],
                'gpp_subgraph': batch[11]
            }

            output = model(inputs)
            loss = output[0]
            losses.append(loss.cpu().item())
            loss.backward()
            
            if step % args.gradient_accumulation_steps == 0:
                if args.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                
                optimizer.step()
                scheduler.step()
                model.zero_grad()
                num_steps += 1
            
            wandb.log({"loss": loss.item()}, step=num_steps)
            
            if step + 1 == len(train_dataloader) - 1:
                evi_f1, evi_r, evi_p = evaluate(args, dev_features, model)
                dev_output = {
                    "dev_evi_F1": evi_f1,
                    "dev_evi_recall": evi_r,
                    "dev_evi_precision": evi_p,
                }
                wandb.log(dev_output, step=num_steps)
                print(f"Epoch {epoch+1} ----- Avg Training Loss: {torch.tensor(losses).mean().item()} | F1: {evi_f1} | Recall: {evi_r} | Precision: {evi_p}")
                if evi_f1 > best_evi_f1:
                    print("Best score!")
                    best_evi_f1 = evi_f1
                    torch.save(model.state_dict(), args.save_path)
                print("----------------------")
                model.train()

def report(args, features, model):
    dataloader = DataLoader(features, batch_size=1, shuffle=False, collate_fn=collate_fn, drop_last=False)
    print("Reporting...")
    model.eval()

    with torch.no_grad():
        preds, hts = {}, {}

        for _, batch in enumerate(tqdm(dataloader, desc='Testing')):
            inputs = {
                'input_ids': batch[0].to(args.device),
                'attention_mask': batch[1].to(args.device),
                'labels': batch[2],
                'entity_pos': batch[3],
                'hts': batch[4],
                'sent_map': batch[5],
                'raw_evidence': batch[6],
                'gss_subgraph': batch[7],
                'pair_evidence': batch[8],
                'triple_evidence': batch[9],
                'gps_subgraph': batch[10],
                'gpp_subgraph': batch[11]
            }
            title = batch[12]
            _, s_logits, _ = model(inputs, report=True)
            pred = torch.sigmoid(s_logits)
            pred = pred.gt(0.5).int().cpu()
            preds[title[0]] = pred
            hts[title[0]] = inputs['hts']

        evi_preds = {k: v[:, :] for k, v in preds.items()}
        return evi_preds


def main():
    parser = argparse.ArgumentParser()

    # datasets path
    parser.add_argument('--train_file', type=str, default='train_annotated.json')
    parser.add_argument('--dev_file', type=str, default='dev.json')
    parser.add_argument('--test_file', type=str, default='test.json')

    parser.add_argument("--data_dir", default="./dataset/docred/", type=str)
    parser.add_argument("--transformer_type", default="bert", type=str)
    parser.add_argument("--model_name_or_path", default="bert-base-cased", type=str)
    parser.add_argument("--save_path", default="./checkpoints/cder.pt", type=str)
    parser.add_argument("--load_path", default="", type=str)

    parser.add_argument("--config_name", default="", type=str, help="Pretrained config name or path if not the same as model_name")
    parser.add_argument("--tokenizer_name", default="", type=str, help="Pretrained tokenizer name or path if not the same as model_name")
    parser.add_argument("--max_seq_length", default=1024, type=int, help="The maximum total input sequence length after tokenization. Sequences longer than this will be truncated, sequences shorter will be padded.")
    parser.add_argument("--num_class", type=int, default=97, help="Number of relation types in dataset.")
    parser.add_argument("--seed", type=int, default=66, help="random seed for initialization")
    parser.add_argument("--train_batch_size", default=4, type=int, help="Batch size for training.")
    parser.add_argument("--test_batch_size", default=4, type=int, help="Batch size for training.")
    parser.add_argument("--grouped_learning_rate", default=1e-4, type=float, help="The initial learning rate for new layers for AdamW.")
    parser.add_argument("--learning_rate", default=5e-5, type=float, help="The initial learning rate for transformer layers for AdamW.")
    parser.add_argument("--adam_epsilon", default=1e-6, type=float, help="Epsilon for AdamW optimizer.")
    parser.add_argument("--num_train_epochs", default=30.0, type=float, help="Total number of training epochs to perform.")
    parser.add_argument("--gradient_accumulation_steps", default=1, type=int, help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--warmup_ratio", default=0.06, type=float, help="Warm up ratio for Adam.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")

    parser.add_argument('--do_test', action='store_true')
    parser.add_argument('--do_infer', action='store_true')

    # Load dataset
    args = parser.parse_args()
    wandb.init(project="CDER")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    args.n_gpu = torch.cuda.device_count()
    args.device = device

    config = AutoConfig.from_pretrained(
        args.config_name if args.config_name else args.model_name_or_path,
        num_labels=args.num_class,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
    )

    read = read_docred

    train_file = os.path.join(args.data_dir, args.train_file)
    dev_file = os.path.join(args.data_dir, args.dev_file)

    train_features = read(train_file, tokenizer, max_seq_length=args.max_seq_length)
    dev_features = read(dev_file, tokenizer, max_seq_length=args.max_seq_length)

    if args.do_infer:
        test_file = os.path.join(args.data_dir, args.test_file)
        test_features = read(test_file, tokenizer, max_seq_length=args.max_seq_length, train=False)

    transformer = AutoModel.from_pretrained(
        args.model_name_or_path,
        from_tf=bool(".ckpt" in args.model_name_or_path),
        config=config,
    ).to(args.device)

    config.cls_token_id = tokenizer.cls_token_id
    config.sep_token_id = tokenizer.sep_token_id
    config.transformer_type = args.transformer_type

    set_all_seeds(args.seed)

    model = CDER(config, transformer)
    model.to(args.device)
    
    if args.load_path == "":    # training
        train(args, model, train_features, dev_features)
    elif args.load_path != "" and args.do_test:     # test
        model.load_state_dict(torch.load(args.load_path))
        evi_f1, evi_r, evi_p = evaluate(args, dev_features, model)
        print(f"Evi-F1 - {evi_f1} | Evi-R - {evi_r} | Evi-P - {evi_p}")
    elif args.load_path != "" and args.do_infer:   # inference
        model.load_state_dict(torch.load(args.load_path))
        evi_preds = report(args, test_features, model)
        with open('./checkpoints/evidence.pkl', 'wb') as f:
            pickle.dump(evi_preds, f)

if __name__ == "__main__":
    main()