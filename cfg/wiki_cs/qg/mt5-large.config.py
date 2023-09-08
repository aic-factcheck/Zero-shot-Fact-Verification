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

    # "regular" mode generates questions for SUPPORTED and REFUTED claims, "nei" mode aims for NEI (extended context)
    QG_MODES = ["regular", "nei"]
    # QG_MODES = ["nei"]

    # highlight = False
    # model_args = ModelArguments(model_name_or_path="/home/drchajan/devel/python/FC/Zero-shot-Fact-Verification/experiments/qg/facebook/mbart-large-cc25_cs_CZ/checkpoint-10000")
    # model_args = ModelArguments(model_name_or_path="/home/drchajan/devel/python/FC/Zero-shot-Fact-Verification/experiments/qg/google/mt5-base_cs_CZ/checkpoint-40000")
    # highlight = True
    # model_args = ModelArguments(model_name_or_path="/home/drchajan/devel/python/FC/Zero-shot-Fact-Verification/experiments/qg/google/mt5-large_cs_CZ_hl/checkpoint-48000")
    MODEL_NAME = f"/home/drchajan/devel/python/FC/Zero-shot-Fact-Verification/experiments/qg/google/mt5-large_{LANG}/checkpoint-59000"
    HIGHLIGHT = False

    DATESTR = datetime.now().strftime("%y%m%d_%H%M%S")
    NOTE = f"Generates Wiki-CS questions for claim generation."

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
        "splits": [
            {"name": "train", "file": Path(NER_ROOT, "train_ners.json"), "size": 10000},
            {"name": "dev", "file": Path(NER_ROOT, "dev_ners.json"), "size": 1000},
            {"name": "test", "file": Path(NER_ROOT, "test_ners.json"), "size": 1000},
        ],
        "highlight": HIGHLIGHT,
        "note": NOTE,
        "date": DATESTR,
    }