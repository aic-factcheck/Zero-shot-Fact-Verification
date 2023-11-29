from datetime import datetime
from pathlib import Path

def config():
    LANG = "cs_CZ"
    LANG_SHORT = "cs"
    
    NLI_ROOT = Path("/mnt/data/factcheck/NLI/csfever_nli_cls")
    PVI_ROOT = Path(NLI_ROOT, "pvi_overfit")

    MODEL_ROOT = "/home/drchajan/devel/python/FC/Zero-shot-Fact-Verification/experiments/nli_fever/"
    MODEL_NAME = Path(MODEL_ROOT, "deepset/xlm-roberta-large-squad2_cs_CZ_lr1e-6/checkpoint-190720")
    NULL_MODEL_NAME = Path(MODEL_ROOT, "deepset/xlm-roberta-large-squad2_cs_CZ_lr1e-6_zero_input/checkpoint-71744")
    MAX_LENGTH = 512

    DATESTR = datetime.now().strftime("%y%m%d_%H%M%S")
    NOTE = f"Computes PVI on LREV CsFEVER-NLI test split and save its extended version. CsFEVER-NLI model (newer than LREV, overfitted)."

    return {
        "lang": LANG,
        "lang_short": LANG_SHORT,
        "pvi_root": PVI_ROOT,
        "model_name": MODEL_NAME,
        "null_model_name": NULL_MODEL_NAME,
        "max_length": MAX_LENGTH,
        "splits": {
            # "train": Path(NLI_ROOT, "train_nli.jsonl"},
            # "dev": Path(NLI_ROOT, "dev_nli.jsonl"),
            "test": str(Path(NLI_ROOT, "test_nli.jsonl")),
        },
        "note": NOTE,
        "date": DATESTR,
    }