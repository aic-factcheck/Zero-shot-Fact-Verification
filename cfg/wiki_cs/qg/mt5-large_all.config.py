from datetime import datetime
from pathlib import Path

def config():
    SEED = 1234

    LANG = "cs_CZ"
    LANG_SHORT = "cs"
    
    DATE = "20230801"

    WIKI_ROOT = f"/mnt/data/factcheck/wiki/{LANG_SHORT}/{DATE}"
    WIKI_CORPUS = f"{WIKI_ROOT}/paragraphs/{LANG_SHORT}wiki-{DATE}-paragraphs.jsonl"

    QACG_ROOT = f"{WIKI_ROOT}/qacg"

    NER_DIR = "PAV-ner-CNEC"
    NER_ROOT = Path(QACG_ROOT, "ner", NER_DIR)

    QG_DIR = "mt5-large_all-cp126k"
    QG_ROOT = Path(QACG_ROOT, "qg", NER_DIR, QG_DIR)

    # "regular" mode generates questions for SUPPORTED and REFUTED claims, "nei" mode aims for NEI (extended context)
    QG_MODES = ["regular", "nei"]
    # QG_MODES = ["nei"]

    MODEL_NAME = "/mnt/personal/drchajan/devel/python/FC/Zero-shot-Fact-Verification/experiments/qg/google/mt5-large_all/checkpoint-126000"
    HIGHLIGHT = False

    DATESTR = datetime.now().strftime("%y%m%d_%H%M%S")
    NOTE = f"Generates Wiki-CS questions for claim generation."

    # we want to have roughly 200k generated regular train samples
    SCALE = 200000/131655

    return {
        "seed": SEED,
        "lang": LANG,
        "lang_short": LANG_SHORT,
        "wiki_root": WIKI_ROOT,
        "qacg_root": QACG_ROOT,
        "ner_root": NER_ROOT,
        "qg_root": QG_ROOT,
        "qg_modes": QG_MODES,
        "model_name": MODEL_NAME,
        "wiki_corpus": WIKI_CORPUS,
        "nei_documents": 2,
        "splits": [
            {"name": "train", "file": Path(NER_ROOT, "train_ners.json"), "size": int(10000*SCALE)},
            {"name": "dev", "file": Path(NER_ROOT, "dev_ners.json"), "size": int(1000*SCALE)},
            {"name": "test", "file": Path(NER_ROOT, "test_ners.json"), "size": int(1000*SCALE)},
        ],
        "highlight": HIGHLIGHT,
        "note": NOTE,
        "date": DATESTR,
    }