from datetime import datetime
from pathlib import Path

def config():
    SEED = 1234

    LANG = "sk_SK"
    LANG_SHORT = "sk"

    DATE = "20230220"
    WIKI_ROOT = f"/mnt/data/factcheck/wiki/{LANG_SHORT}/{DATE}"
    
    QACG_ROOT = f"{WIKI_ROOT}/qacg"

    NER_DIR = "crabz_slovakbert-ner"
    NER_ROOT = Path(QACG_ROOT, "ner", NER_DIR)
    
    QG_DIR = "mt5-large-cp37k"
    QG_ROOT = Path(QACG_ROOT, "qg", NER_DIR, QG_DIR)

    MODEL_NAME = "/home/drchajan/devel/python/FC/Zero-shot-Fact-Verification/experiments/qg/google/mt5-large_sk_SK/checkpoint-37000"
    HIGHLIGHT = False

    WIKI_CORPUS = f"{WIKI_ROOT}/paragraphs/{LANG_SHORT}wiki-{DATE}-paragraphs.jsonl"

    DATESTR = datetime.now().strftime("%y%m%d_%H%M%S")
    NOTE = f"Generates Wiki-SK questions for claim generation."

    return {
        "seed": SEED,
        "lang": LANG,
        "lang_short": LANG_SHORT,
        "wiki_root": WIKI_ROOT,
        "qacg_root": QACG_ROOT,
        "ner_root": NER_ROOT,
        "qg_root": QG_ROOT,
        "model_name": MODEL_NAME,
        "corpus": WIKI_CORPUS,
        "splits": [
            {"name": "train", "file": Path(NER_ROOT, "train_ners.json"), "size": 10000},
            {"name": "dev", "file": Path(NER_ROOT, "dev_ners.json"), "size": 1000},
            {"name": "test", "file": Path(NER_ROOT, "test_ners.json"), "size": 1000},
        ],
        "highlight": HIGHLIGHT,
        "note": NOTE,
        "date": DATESTR,
    }