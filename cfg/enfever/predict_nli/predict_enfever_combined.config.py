from datetime import datetime
from pathlib import Path

def config():
    NAME = "enfever_combined"
    LANG = "en_US"
    LANG_SHORT = "en"
    
    NLI_ROOT = Path("/mnt/data/factcheck/NLI/nli_fever_cls")

    MODEL_ROOT = "/home/drchajan/devel/python/FC/Zero-shot-Fact-Verification/experiments/nli/"
    MODEL_NAME = Path(MODEL_ROOT, "deepset/xlm-roberta-large-squad2_cs_en_pl_sk-20230801_balanced_lr1e-6/checkpoint-207232")
    MAX_LENGTH = 512

    DATESTR = datetime.now().strftime("%y%m%d_%H%M%S")
    NOTE = f"Predicts NLI on EnFEVER-NLI test split."

    return {
        "name": NAME,
        "lang": LANG,
        "lang_short": LANG_SHORT,
        "nli_root": NLI_ROOT,
        "model_name": MODEL_NAME,
        "max_length": MAX_LENGTH,
        "splits": {
            # "train": Path(NLI_ROOT, "train_nli.jsonl"},
            # "dev": Path(NLI_ROOT, "dev_nli.jsonl"),
            "test": str(Path(NLI_ROOT, "test_nli.jsonl")),
        },
        "note": NOTE,
        "date": DATESTR,
    }