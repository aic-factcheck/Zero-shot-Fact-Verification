from datetime import datetime
from pathlib import Path

def config():
    LANG = "en_US"
    LANG_SHORT = "en"
    
    NLI_ROOT = Path("/mnt/data/factcheck/NLI/nli_fever_cls")

    MODEL_ROOT = "/home/drchajan/devel/python/FC/Zero-shot-Fact-Verification/experiments/nli_fever/"
    MODEL_NAME = Path(MODEL_ROOT, "deepset/xlm-roberta-large-squad2_en_US_lr1e-6_zero_input/checkpoint-256")
    MAX_LENGTH = 512

    DATESTR = datetime.now().strftime("%y%m%d_%H%M%S")
    NOTE = f"Calibrates EN language zero-input NLI model."

    return {
        "lang": LANG,
        "lang_short": LANG_SHORT,
        "nli_root": NLI_ROOT,
        "model_name": MODEL_NAME,
        "max_length": MAX_LENGTH,
        "epochs": 50,
        "split": str(Path(NLI_ROOT, "dev_nli.jsonl")),
        "note": NOTE,
        "date": DATESTR,
    }