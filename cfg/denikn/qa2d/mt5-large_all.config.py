from datetime import datetime
from pathlib import Path

def config():
    SEED = 1234

    LANG = "cs_CZ"
    LANG_SHORT = "cs"
    
    DATA_ROOT = f"/mnt/data/factcheck/denikn/v1"
    DATA_CORPUS = Path(DATA_ROOT, "interim", "denikn_paragraphs.jsonl")
    QACG_ROOT = Path(DATA_ROOT, "qacg")

    NER_DIR = "PAV-ner-CNEC"
    NER_ROOT = Path(QACG_ROOT, "ner", NER_DIR)

    QG_DIR = "mt5-large_all-cp126k"
    QG_ROOT = Path(QACG_ROOT, "qg", NER_DIR, QG_DIR)

    MODEL_NAME = f"/mnt/personal/drchajan/devel/python/FC/Zero-shot-Fact-Verification/experiments/qa2d/google/mt5-large_all/BKP/checkpoint-156000"
    MODEL_SHORT = "mt5-large_all-cp156k"

    CLAIM_ROOT = Path(QACG_ROOT, "claim", NER_DIR, QG_DIR, MODEL_SHORT)
    CLAIM_TYPES = ["support", "refute"]

    DATESTR = datetime.now().strftime("%y%m%d_%H%M%S")
    NOTE = f"Generates DenikN claims."

    return {
        "seed": SEED,
        "lang": LANG,
        "lang_short": LANG_SHORT,
        "data_root": DATA_ROOT,
        "qacg_root": QACG_ROOT,
        "ner_root": NER_ROOT,
        "qg_root": QG_ROOT,
        "claim_root": CLAIM_ROOT,
        "claim_types": CLAIM_TYPES,
        "model_name": MODEL_NAME,
        "corpus": DATA_CORPUS,
        "nei_documents": 2,
        "splits": [
            {"name": "train", "file": Path(NER_ROOT, "train_ners.json"), "size": 20000},
            {"name": "dev", "file": Path(NER_ROOT, "dev_ners.json"), "size": 2000},
            {"name": "test", "file": Path(NER_ROOT, "test_ners.json"), "size": 2000},
        ],
        "note": NOTE,
        "date": DATESTR,
    }