from datetime import datetime
from pathlib import Path

def config():
    LANG = "sk_SK"
    LANG_SHORT = "sk"
    
    DATE = "20230801"
    WIKI_ROOT = f"/mnt/data/factcheck/wiki/{LANG_SHORT}/{DATE}"

    QACG_ROOT = f"{WIKI_ROOT}/qacg"

    NER_DIR = "crabz_slovakbert-ner"
    QG_DIR = "mt5-large_all-cp126k"
    QACG_DIR = "mt5-large_all-cp156k"

    NLI_DIR = Path("nli", NER_DIR, QG_DIR, QACG_DIR)
    NLI_ROOT = Path(QACG_ROOT, NLI_DIR)
    PVI_ROOT = Path(QACG_ROOT, NLI_DIR, "pvi_overfit")

    MODEL_ROOT = "/home/drchajan/devel/python/FC/Zero-shot-Fact-Verification/experiments/nli/"
    MODEL_NAME = Path(MODEL_ROOT, "deepset/xlm-roberta-large-squad2_sk_SK-20230801_balanced_lr1e-6/checkpoint-202112")
    NULL_MODEL_NAME = Path(MODEL_ROOT, "deepset/xlm-roberta-large-squad2_sk_SK-20230801_balanced_lr1e-6_zero_input/checkpoint-36480")
    MAX_LENGTH = 512

    DATESTR = datetime.now().strftime("%y%m%d_%H%M%S")
    NOTE = f"Computes PVI on Wiki-SK test split and save its extended version. QACG-SK overfitted model."

    return {
        "lang": LANG,
        "lang_short": LANG_SHORT,
        "pvi_root": PVI_ROOT,
        "model_name": MODEL_NAME,
        "null_model_name": NULL_MODEL_NAME,
        "max_length": MAX_LENGTH,
        "splits": {
            # "train": Path(NLI_ROOT, "train_balanced.jsonl"},
            # "dev": Path(NLI_ROOT, "dev_balanced.jsonl"),
            "test": str(Path(NLI_ROOT, "test_balanced.jsonl")),
        },
        "note": NOTE,
        "date": DATESTR,
    }