from datetime import datetime
from pathlib import Path

def config():
    LANG = "csnews"
    LANG_SHORT = "csnews"
    
    DATA_ROOT = f"/mnt/data/factcheck/qacg/news_sum"
    QACG_ROOT = f"{DATA_ROOT}/qacg"

    NLI_DIR = Path("nli")
    NLI_ROOT = Path(QACG_ROOT, NLI_DIR)

    MODEL_ROOT = "/home/drchajan/devel/python/FC/Zero-shot-Fact-Verification/experiments/nli/"
    MODEL_NAME = Path(MODEL_ROOT, "deepset/xlm-roberta-large-squad2_csnews-_balanced_shuf_lr1e-6/checkpoint-60544")
    MAX_LENGTH = 512

    DATESTR = datetime.now().strftime("%y%m%d_%H%M%S")
    NOTE = f"Calibrates CSNEWS language NLI model."

    return {
        "lang": LANG,
        "lang_short": LANG_SHORT,
        "data_root": DATA_ROOT,
        "nli_root": NLI_ROOT,
        "model_name": MODEL_NAME,
        "max_length": MAX_LENGTH,
        "epochs": 50,
        "split": str(Path(NLI_ROOT, "dev_balanced_shuf.jsonl")),
        "note": NOTE,
        "date": DATESTR,
    }