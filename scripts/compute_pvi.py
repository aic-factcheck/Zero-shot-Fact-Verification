from collections import Counter, OrderedDict
from pathlib import Path
from tqdm import tqdm

import numpy as np
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

from datasets import load_dataset
from transformers import (
    AutoConfig,
    AutoModelForSeq2SeqLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
)

from aic_nlp_utils.json import read_jsonl, read_json, write_json, write_jsonl
from aic_nlp_utils.pycfg import parse_pycfg_args, read_pycfg

from zshot_fact_verify.models.load import load_tokenizer_and_model, find_last_checkpoint
from zshot_fact_verify.models.inference import split_predict
from zshot_fact_verify.wiki.load import load_corpus, create_corpus_splits, select_nei_context_for_splits, load_nei_ners
from zshot_fact_verify.qa2d.qa2d import SameDocumentNERReplacementGenerator, ClaimGenerator


def main ():
    args = parse_pycfg_args()

    def save_dir_fn(cfg):
        return Path(cfg["pvi_root"], f"compute_pvi.config.py")
    
    cfg = read_pycfg(args.pycfg, save_dir_fn=save_dir_fn)

    model_name = cfg["model_name"]
    model0_name = cfg["null_model_name"]

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"model name: '{model_name}'")
    print(f"∅ model name: '{model0_name}'")
    print(f"device: {device}")
    print(f"splits =\n{cfg['splits']}")

    raw_nli = load_dataset("json", data_files=cfg["splits"])

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    id2label = cfg.get("id2label", None)
    if id2label:
        label2id = {v: k for k,v in id2label.items()}
        model = AutoModelForSequenceClassification.from_pretrained(model_name, device_map="auto", id2label=id2label, label2id=label2id)
        model0 = AutoModelForSequenceClassification.from_pretrained(model0_name, device_map="auto", id2label=id2label, label2id=label2id)

    else:
        model = AutoModelForSequenceClassification.from_pretrained(model_name, device_map="auto")
        model0 = AutoModelForSequenceClassification.from_pretrained(model0_name, device_map="auto")


    for split_name in raw_nli:
        P, _, T = split_predict(model, tokenizer, raw_nli[split_name], device=device, max_length=cfg["max_length"], 
                                apply_softmax=True, one_hot_targets=True, to_numpy=True)
        P0, _, _ = split_predict(model0, tokenizer, raw_nli[split_name], device=device, max_length=cfg["max_length"],
                                   apply_softmax=True, one_hot_targets=True, to_numpy=True)
        
        print(f"P.shape={P.shape}, T.shape={T.shape}")
        logPtarget = np.log2(P[np.arange(len(T)), T])
        logP0target = np.log2(P0[np.arange(len(T)), T])
        print(f"logPtarget.shape={logPtarget.shape}")

        PVIs = -logP0target + logPtarget
        VUI = np.mean(PVIs)
        print(f"V-usable information: {VUI}")

        src_split = read_jsonl(cfg["splits"][split_name])
        assert len(PVIs) == len(src_split), (len(PVIs), len(src_split)) 
        for sample, pvi in zip(src_split, PVIs):
            sample["PVI"] = pvi

        out_file = Path(cfg["pvi_root"], Path(cfg["splits"][split_name]).name)
        print(f"saving '{split_name}' split with PVI to '{out_file}'")
        write_jsonl(out_file, src_split, mkdir=True)


if __name__ == "__main__":
    main()