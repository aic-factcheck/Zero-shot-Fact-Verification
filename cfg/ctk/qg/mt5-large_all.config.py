from datetime import datetime
from pathlib import Path

def config():
    SEED = 1234

    LANG = "cs_CZ"
    LANG_SHORT = "cs"
    
    DATA_ROOT = f"/mnt/data/ctknews/factcheck/par6"
    DATA_CORPUS = Path(DATA_ROOT, "interim", "jsonl", "ctk_filtered.jsonl")
    QACG_ROOT = Path(DATA_ROOT, "qacg")

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
    NOTE = f"Generates CTK questions for claim generation."

    return {
        "seed": SEED,
        "lang": LANG,
        "lang_short": LANG_SHORT,
        "data_root": DATA_ROOT,
        "qacg_root": QACG_ROOT,
        "ner_root": NER_ROOT,
        "qg_root": QG_ROOT,
        "qg_modes": QG_MODES,
        "model_name": MODEL_NAME,
        "corpus": DATA_CORPUS,
        "nei_documents": 2,
        "splits": [
            {"name": "train", "file": Path(NER_ROOT, "train_ners.json"), "size": 50000},
            {"name": "dev", "file": Path(NER_ROOT, "dev_ners.json"), "size": 5000},
            {"name": "test", "file": Path(NER_ROOT, "test_ners.json"), "size": 5000},
        ],
        "highlight": HIGHLIGHT,
        "note": NOTE,
        "date": DATESTR,
    }