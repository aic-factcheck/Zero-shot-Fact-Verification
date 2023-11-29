from datetime import datetime
from pathlib import Path

def config():
    SEED = 1234

    LANG = "cs_CZ"
    LANG_SHORT = "cs"
    
    # DATE = "20230220"
    DATE = "20230801"
    WIKI_ROOT = f"/mnt/data/factcheck/wiki/{LANG_SHORT}/{DATE}"
    WIKI_CORPUS = f"{WIKI_ROOT}/paragraphs/{LANG_SHORT}wiki-{DATE}-paragraphs.jsonl"

    QACG_ROOT = f"{WIKI_ROOT}/qacg"

    NER_DIR = "PAV-ner-CNEC"
    NER_ROOT = Path(QACG_ROOT, "ner", NER_DIR)

    QG_DIR = "mt5-large-cp59k"
    QG_ROOT = Path(QACG_ROOT, "qg", NER_DIR, QG_DIR)

    MODEL_NAME = f"/home/drchajan/devel/python/FC/Zero-shot-Fact-Verification/experiments/qa2d/facebook/mbart-large-cc25_{LANG}/checkpoint-26000"
    MODEL_SHORT = "mbart-large-cc25_cp26k"

    CLAIM_ROOT = Path(QACG_ROOT, "claim", NER_DIR, QG_DIR, MODEL_SHORT)
    CLAIM_TYPES = ["support", "refute", "nei"]

    DATESTR = datetime.now().strftime("%y%m%d_%H%M%S")
    NOTE = f"Generates Wiki-CS claims."

    return {
        "seed": SEED,
        "lang": LANG,
        "lang_short": LANG_SHORT,
        "wiki_root": WIKI_ROOT,
        "qacg_root": QACG_ROOT,
        "ner_root": NER_ROOT,
        "qg_root": QG_ROOT,
        "claim_root": CLAIM_ROOT,
        "claim_types": CLAIM_TYPES,
        "model_name": MODEL_NAME,
        "corpus": WIKI_CORPUS,
        "splits": [
            {"name": "train", "file": Path(NER_ROOT, "train_ners.json"), "size": 10000},
            {"name": "dev", "file": Path(NER_ROOT, "dev_ners.json"), "size": 1000},
            {"name": "test", "file": Path(NER_ROOT, "test_ners.json"), "size": 1000},
        ],
        "note": NOTE,
        "date": DATESTR,
    }