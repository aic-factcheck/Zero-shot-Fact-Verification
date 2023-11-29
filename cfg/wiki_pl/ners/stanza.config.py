from datetime import datetime
from pathlib import Path

def config():
    SEED = 1234
    LANG = "pl_PL"
    LANG_SHORT = "pl"
    METHOD = "stanza"
    NER_DIR = "stanza"

    # DATE = "20230220"
    DATE = "20230801"
    WIKI_ROOT = f"/mnt/data/factcheck/wiki/{LANG_SHORT}/{DATE}"
    QACG_ROOT = f"{WIKI_ROOT}/qacg"
    NER_ROOT = Path(QACG_ROOT, "ner", NER_DIR)

    WIKI_CORPUS = f"{WIKI_ROOT}/paragraphs/{LANG_SHORT}wiki-{DATE}-paragraphs.jsonl"

    DATESTR = datetime.now().strftime("%y%m%d_%H%M%S")
    NOTE = f"Generates Wiki-PL-based NER data for claim generation."

    # we want to have roughly 200k generated regular train samples
    SCALE = 200000/140372
    
    return {
        "seed": SEED,
        "lang": LANG,
        "lang_short": LANG_SHORT,
        "method": METHOD,
        "wiki_root": WIKI_ROOT,
        "qacg_root": QACG_ROOT,
        "ner_root": NER_ROOT,
        "corpus": WIKI_CORPUS,
        "nei_documents": 2,
        "splits": [
            {"name": "train", "file": Path(NER_ROOT, "train_ners.json"), "size": int(10000*SCALE)},
            {"name": "dev", "file": Path(NER_ROOT, "dev_ners.json"), "size": int(1000*SCALE)},
            {"name": "test", "file": Path(NER_ROOT, "test_ners.json"), "size": int(1000*SCALE)},
        ],
        "note": NOTE,
        "date": DATESTR,
    }