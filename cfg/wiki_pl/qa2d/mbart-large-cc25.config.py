from datetime import datetime
from pathlib import Path

def config():
    SEED = 1234
    
    LANG = "pl_PL"
    LANG_SHORT = "pl"

    DATE = "20230220"
    WIKI_ROOT = f"/mnt/data/factcheck/wiki/{LANG_SHORT}/{DATE}"
    WIKI_CORPUS = f"{WIKI_ROOT}/paragraphs/{LANG_SHORT}wiki-{DATE}-paragraphs.jsonl"
    
    QACG_ROOT = f"{WIKI_ROOT}/qacg"

    NER_DIR = "stanza"
    NER_ROOT = Path(QACG_ROOT, "ner", NER_DIR)
    
    QG_DIR = "mt5-large-cp34k"
    QG_ROOT = Path(QACG_ROOT, "qg", NER_DIR, QG_DIR)

    MODEL_NAME = f"/home/drchajan/devel/python/FC/Zero-shot-Fact-Verification/experiments/qa2d/facebook/mbart-large-cc25_{LANG}/checkpoint-43000"
    MODEL_SHORT = "mbart-large-cc25_cp43k"
    CLAIM_ROOT = Path(QACG_ROOT, "claim", NER_DIR, QG_DIR, MODEL_SHORT)

    DATESTR = datetime.now().strftime("%y%m%d_%H%M%S")
    NOTE = f"Generates Wiki-PL claims."

    return {
        "seed": SEED,
        "lang": LANG,
        "lang_short": LANG_SHORT,
        "wiki_root": WIKI_ROOT,
        "qacg_root": QACG_ROOT,
        "ner_root": NER_ROOT,
        "qg_root": QG_ROOT,
        "claim_root": CLAIM_ROOT,
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