from collections import Counter, OrderedDict
from pathlib import Path
from tqdm import tqdm

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
        return Path(cfg["model_name"], f"predict_nli_{cfg['name']}.config.py")
    
    cfg = read_pycfg(args.pycfg, save_dir_fn=save_dir_fn)

    model_name = cfg["model_name"]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"  model name: '{model_name}'")
    print(f"device: {device}")
    print(f"splits =\n{cfg['splits']}")

    raw_nli = load_dataset("json", data_files=cfg["splits"])

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    id2label = cfg.get("id2label", None)
    if id2label:
        label2id = {v: k for k,v in id2label.items()}
        model = AutoModelForSequenceClassification.from_pretrained(model_name, device_map="auto", id2label=id2label, label2id=label2id)
    else:
        model = AutoModelForSequenceClassification.from_pretrained(model_name, device_map="auto")

    for split_name in raw_nli:
        Y, C, T = split_predict(model, tokenizer, raw_nli[split_name], device=device, max_length=cfg["max_length"])

        accuracy = accuracy_score(T, C)
        f1_macro = f1_score(T, C, average='macro')
        cmatrix = confusion_matrix(T, C, labels=['s', 'r', 'n'])
        print(f"accuracy: {accuracy}")
        print(f"F1 macro: {f1_macro}")
        print()
        print(f"confusion matrix:\n{cmatrix}")

        preds = {
            "accuracy": accuracy,
            "f1_macro": f1_macro,
            "cmatrix": cmatrix.tolist(),
            "pred_classes": C,
            "targets": T,
            "pred_probs": Y.tolist(),
        }
        report_file = Path(model_name, f"predictions_{cfg['name']}_{split_name}.json")
        print(f"saving predictions for '{split_name}' to '{report_file}'")
        write_json(report_file, preds)


if __name__ == "__main__":
    main()