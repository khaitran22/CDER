import torch
from torch.utils.data import DataLoader
from utils import collate_fn
from tqdm import tqdm
import numpy as np

def evaluate(args, features, model):
    dataloader = DataLoader(features, batch_size=args.test_batch_size, shuffle=False, collate_fn=collate_fn, drop_last=False)
    print("Evaluating...")
    model.eval()
    
    with torch.no_grad():
        preds, labels = [], []
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
                'gpp_subgraph': batch[11],
                'title': batch[12]
            }

            _, s_logits, label = model(inputs)
            pred = torch.sigmoid(s_logits)
            pred = pred.gt(0.5).int().cpu()
            preds.append(pred)
            labels.append(label)
        
        preds = torch.cat(preds, dim=0)
        labels = torch.cat(labels, dim=0)
        total_pred = preds.sum(); ground_truth = labels.sum(); correct_pred = (preds * labels).sum()
        
        evi_r = correct_pred / ground_truth

        if total_pred > 0:
            evi_p = 1.0 * correct_pred / total_pred
        else:
            evi_p = 0

        if evi_p + evi_r == 0:
            evi_f1 = 0
        else:
            evi_f1 = 2.0 * evi_p * evi_r / (evi_p + evi_r)

        evi_f1 = np.around(evi_f1*100, decimals=2).item()
        evi_r = np.around(evi_r*100, decimals=2).item()
        evi_p = np.around(evi_p*100, decimals=2).item()

        return evi_f1, evi_r, evi_p