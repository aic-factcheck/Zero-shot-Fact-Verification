from datetime import datetime
from pathlib import Path

def config():
    SEED = 1234

    LANG = "sk_SK"
    LANG_SHORT = "sk"
    
    DATE = "20230801"
    WIKI_ROOT = f"/mnt/data/factcheck/wiki/{LANG_SHORT}/{DATE}"
    WIKI_CORPUS = f"{WIKI_ROOT}/paragraphs/{LANG_SHORT}wiki-{DATE}-paragraphs.jsonl"

    QACG_ROOT = f"{WIKI_ROOT}/qacg"

    NER_DIR = "crabz_slovakbert-ner"
    NER_ROOT = Path(QACG_ROOT, "ner", NER_DIR)
    
    QG_DIR = "mt5-large_all-cp126k"
    QG_ROOT = Path(QACG_ROOT, "qg", NER_DIR, QG_DIR)

    MODEL_NAME = f"/mnt/personal/drchajan/devel/python/FC/Zero-shot-Fact-Verification/experiments/qa2d/google/mt5-large_all/BKP/checkpoint-130000"
    MODEL_SHORT = "mt5-large_all-cp130k"

    CLAIM_ROOT = Path(QACG_ROOT, "claim", NER_DIR, QG_DIR, MODEL_SHORT)
    CLAIM_TYPES = ["support", "refute", "nei"]

    DATESTR = datetime.now().strftime("%y%m%d_%H%M%S")
    NOTE = f"Generates Wiki-SK claims."

    # we want to have roughly 200k generated regular train samples
    SCALE = 200000/114582
    
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
        "wiki_corpus": WIKI_CORPUS,
        "nei_documents": 2,
        "splits": [
            {"name": "train", "file": Path(NER_ROOT, "train_ners.json"), "size": int(10000*SCALE)},
            {"name": "dev", "file": Path(NER_ROOT, "dev_ners.json"), "size": int(1000*SCALE)},
            {"name": "test", "file": Path(NER_ROOT, "test_ners.json"), "size": int(1000*SCALE)},
        ],
        "note": NOTE,
        "date": DATESTR,
    }