from datetime import datetime
from pathlib import Path

def config():
    NAME = "wiki-en"
    LANG = "en_US"
    LANG_SHORT = "en"
    
    DATE = "20230801"
    WIKI_ROOT = f"/mnt/data/factcheck/wiki/{LANG_SHORT}/{DATE}"

    QACG_ROOT = f"{WIKI_ROOT}/qacg"

    NER_DIR = "stanza"
    QG_DIR = "mt5-large_all-cp126k"
    QACG_DIR = "mt5-large_all-cp156k"

    NLI_DIR = Path("nli", NER_DIR, QG_DIR, QACG_DIR)
    NLI_ROOT = Path(QACG_ROOT, NLI_DIR)

    MODEL_NAME = Path("/mnt/data/factcheck/nli_models/lrev_aug22/paper/enfever_nli/deepset/xlm-roberta-large-squad2_bs10_ep31_wr0.2")
    MAX_LENGTH = 512

    ID2LABEL = {0: "s", 1: "r", 2: "n"}

    DATESTR = datetime.now().strftime("%y%m%d_%H%M%S")
    NOTE = f"Predicts LREV NLI model on Wiki-EN test split."

    return {
        "name": NAME,
        "lang": LANG,
        "lang_short": LANG_SHORT,
        "wiki_root": WIKI_ROOT,
        "nli_root": NLI_ROOT,
        "model_name": MODEL_NAME,
        "max_length": MAX_LENGTH,
        "id2label": ID2LABEL,
        "splits": {
            # "train": Path(NLI_ROOT, "train_balanced.jsonl"},
            # "dev": Path(NLI_ROOT, "dev_balanced.jsonl"),
            "test": str(Path(NLI_ROOT, "test_balanced.jsonl")),
        },
        "note": NOTE,
        "date": DATESTR,
    }