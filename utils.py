import json
import logging
import os
import random
from pathlib import Path

import dataclasses
import numpy as np
import torch


def set_all_seeds(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def setup_logging():
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO)


def parse_config(args_class, json_file):
    data = json.loads(Path(json_file).read_text())
    keys = {f.name for f in dataclasses.fields(args_class)}
    inputs = {k: v for k, v in data.items() if k in keys}
    return args_class(**inputs)

def collate_fn(batch):
    max_len = max([len(f["input_ids"]) for f in batch])
    input_ids = [f["input_ids"] + [0] *
                 (max_len - len(f["input_ids"])) for f in batch]
    input_mask = [[1.0] * len(f["input_ids"]) + [0.0]
                  * (max_len - len(f["input_ids"])) for f in batch]
    labels = [f["labels"] for f in batch]
    entity_pos = [f["entity_pos"] for f in batch]
    hts = [f["hts"] for f in batch]
    sent_map = [f["sent_map"] for f in batch]
    raw_evidence = [f["raw_evidence"] for f in batch]
    pair_evidence = [f["pair_evidence"] for f in batch]
    triple_evidence = [f["triple_evidence"] for f in batch]
    title = [f["title"] for f in batch]
    gss_subgraph = [f["gss_subgraph"] for f in batch]
    gps_subgraph = [f["gps_subgraph"] for f in batch]
    gpp_subgraph = [f["gpp_subgraph"] for f in batch]
    input_ids = torch.tensor(input_ids, dtype=torch.long)
    input_mask = torch.tensor(input_mask, dtype=torch.float)
    output = (input_ids, input_mask, labels, entity_pos, hts, sent_map, raw_evidence, gss_subgraph, pair_evidence, triple_evidence, gps_subgraph, gpp_subgraph, title)
    return output