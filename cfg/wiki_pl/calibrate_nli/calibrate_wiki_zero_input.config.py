from datetime import datetime
from pathlib import Path

def config():
    LANG = "pl_PL"
    LANG_SHORT = "pl"
    
    DATE = "20230801"
    WIKI_ROOT = f"/mnt/data/factcheck/wiki/{LANG_SHORT}/{DATE}"
    QACG_ROOT = f"{WIKI_ROOT}/qacg"

    NER_DIR = "stanza"
    QG_DIR = "mt5-large_all-cp126k"
    QACG_DIR = "mt5-large_all-cp156k"

    NLI_DIR = Path("nli", NER_DIR, QG_DIR, QACG_DIR)
    NLI_ROOT = Path(QACG_ROOT, NLI_DIR)

    MODEL_ROOT = "/home/drchajan/devel/python/FC/Zero-shot-Fact-Verification/experiments/nli/"
    MODEL_NAME = Path(MODEL_ROOT, "deepset/xlm-roberta-large-squad2_pl_PL-20230801_balanced_lr1e-6_zero_input/checkpoint-8992")
    MAX_LENGTH = 512

    DATESTR = datetime.now().strftime("%y%m%d_%H%M%S")
    NOTE = f"Calibrates PL language zero-input NLI model."

    return {
        "lang": LANG,
        "lang_short": LANG_SHORT,
        "wiki_root": WIKI_ROOT,
        "nli_root": NLI_ROOT,
        "model_name": MODEL_NAME,
        "max_length": MAX_LENGTH,
        "epochs": 50,
        "split": str(Path(NLI_ROOT, "dev_balanced.jsonl")),
        "note": NOTE,
        "date": DATESTR,
    }