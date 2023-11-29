from datetime import datetime
from pathlib import Path

def config():
    SEED = 1234
    LANG = "en_US"
    LANG_SHORT = "en"
    METHOD = "stanza"
    NER_DIR = "stanza"

    # DATE = "20230220"
    DATE = "20230801"
    WIKI_ROOT = f"/mnt/data/factcheck/wiki/{LANG_SHORT}/{DATE}"
    QACG_ROOT = f"{WIKI_ROOT}/qacg"
    NER_ROOT = Path(QACG_ROOT, "ner", NER_DIR)

    WIKI_CORPUS = f"{WIKI_ROOT}/paragraphs/{LANG_SHORT}wiki-{DATE}-paragraphs.jsonl"

    DATESTR = datetime.now().strftime("%y%m%d_%H%M%S")
    NOTE = f"Generates Wiki-EN NERs for claim generation."

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
            {"name": "train", "file": Path(NER_ROOT, "train_ners.json"), "size": 10000},
            {"name": "dev", "file": Path(NER_ROOT, "dev_ners.json"), "size": 1000},
            {"name": "test", "file": Path(NER_ROOT, "test_ners.json"), "size": 1000},
        ],
        "note": NOTE,
        "date": DATESTR,
    }