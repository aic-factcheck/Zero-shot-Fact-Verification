from datetime import datetime
from pathlib import Path

def config():
    LANG = "cs_CZ"
    LANG_SHORT = "cs"
    
    NLI_ROOT = Path("/mnt/data/factcheck/NLI/csfever_nli_cls")

    MODEL_ROOT = "/home/drchajan/devel/python/FC/Zero-shot-Fact-Verification/experiments/nli_fever/"
    MODEL_NAME = Path(MODEL_ROOT, "deepset/xlm-roberta-large-squad2_cs_CZ_lr1e-6/checkpoint-190720")
    MAX_LENGTH = 512

    DATESTR = datetime.now().strftime("%y%m%d_%H%M%S")
    NOTE = f"Calibrates CS language NLI model."

    return {
        "lang": LANG,
        "lang_short": LANG_SHORT,
        # "wiki_root": WIKI_ROOT,
        "nli_root": NLI_ROOT,
        "model_name": MODEL_NAME,
        "max_length": MAX_LENGTH,
        "epochs": 50,
        "split": str(Path(NLI_ROOT, "dev_nli.jsonl")),
        "note": NOTE,
        "date": DATESTR,
    }