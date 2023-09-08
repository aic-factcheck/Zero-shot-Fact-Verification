from datetime import datetime
from pathlib import Path

def config():
    SEED = 1234

    LANG = "cs_CZ"
    LANG_SHORT = "cs"
    
    METHOD = "transformer_czert"
    # models from https://github.com/kiv-air/Czert
    MODEL_NAME = "/mnt/data/factcheck/models/czert/PAV-ner-CNEC" 
    MODEL_SHORT = "PAV-ner-CNEC"

    NER_DIR = MODEL_SHORT

    # DATE = "20230220"
    DATE = "20230801"
    WIKI_ROOT = f"/mnt/data/factcheck/wiki/{LANG_SHORT}/{DATE}"
    QACG_ROOT = f"{WIKI_ROOT}/qacg"
    NER_ROOT = Path(QACG_ROOT, "ner", NER_DIR)

    WIKI_CORPUS = f"{WIKI_ROOT}/paragraphs/{LANG_SHORT}wiki-{DATE}-paragraphs.jsonl"

    DATESTR = datetime.now().strftime("%y%m%d_%H%M%S")
    NOTE = f"Generates Wiki-CS NERs aimed for claim generation."
    
    # we want to have roughly 200k generated regular train samples
    SCALE = 200000/131655

    return {
        "seed": SEED,
        "lang": LANG,
        "lang_short": LANG_SHORT,
        "method": METHOD,
        "model_name": MODEL_NAME,
        "model_short": MODEL_SHORT,
        "wiki_root": WIKI_ROOT,
        "qacg_root": QACG_ROOT,
        "ner_root": NER_ROOT,
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