from datetime import datetime
from pathlib import Path

def config():
    LANG = "sum_cs_en_pl_sk"
    LANG_SHORT = "sum_cs_en_pl_sk"
    
    DATE = "20230801"
    WIKI_ROOT = f"/mnt/data/factcheck/wiki/{LANG_SHORT}/{DATE}"

    QACG_ROOT = f"{WIKI_ROOT}/qacg"

    NLI_DIR = Path("nli")
    NLI_ROOT = Path(QACG_ROOT, NLI_DIR)
    PVI_ROOT = Path(QACG_ROOT, NLI_DIR, "pvi")

    MODEL_ROOT = "/home/drchajan/devel/python/FC/Zero-shot-Fact-Verification/experiments/nli/"
    MODEL_NAME = Path(MODEL_ROOT, "deepset/xlm-roberta-large-squad2_sum_cs_en_pl_sk-20230801_balanced_lr1e-6/checkpoint-321184")
    NULL_MODEL_NAME = Path(MODEL_ROOT, "deepset/xlm-roberta-large-squad2_sum_cs_en_pl_sk-20230801_balanced_lr1e-6_zero_input/checkpoint-928")
    MAX_LENGTH = 512

    DATESTR = datetime.now().strftime("%y%m%d_%H%M%S")
    NOTE = f"Computes PVI on Wiki-SUM test split and save its extended version. QACG-SUM model."

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