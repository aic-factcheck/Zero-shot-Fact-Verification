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

    DATA_ROOT = f"/mnt/data/cro/factcheck/v1"
    DATA_CORPUS = Path(DATA_ROOT, "interim", "cro_paragraphs_filtered.jsonl")
    QACG_ROOT = Path(DATA_ROOT, "qacg")
    NER_ROOT = Path(QACG_ROOT, "ner", NER_DIR)

    DATESTR = datetime.now().strftime("%y%m%d_%H%M%S")
    NOTE = f"Generates cRO NERs aimed for claim generation."
    
    return {
        "seed": SEED,
        "lang": LANG,
        "lang_short": LANG_SHORT,
        "method": METHOD,
        "model_name": MODEL_NAME,
        "model_short": MODEL_SHORT,
        "data_root": DATA_ROOT,
        "qacg_root": QACG_ROOT,
        "ner_root": NER_ROOT,
        "corpus": DATA_CORPUS,
        "nei_documents": 2,
        "splits": [
            {"name": "train", "file": Path(NER_ROOT, "train_ners.json"), "size": 20000},
            {"name": "dev", "file": Path(NER_ROOT, "dev_ners.json"), "size": 2000},
            {"name": "test", "file": Path(NER_ROOT, "test_ners.json"), "size": 2000},
        ],
        "note": NOTE,
        "date": DATESTR,
    }