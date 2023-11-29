from datetime import datetime
from pathlib import Path

def config():
    NAME = "csfever_lrev"
    LANG = "cs_CZ"
    LANG_SHORT = "cs"
    
    NLI_ROOT = Path("/mnt/data/factcheck/NLI/csfever_nli_cls")

    MODEL_NAME = Path("/mnt/data/factcheck/nli_models/lrev_aug22/paper/csfever_nearestp/xlm-roberta-large-squad2_bs10_ep20_wr0.2")
    MAX_LENGTH = 512

    ID2LABEL = {0: "s", 1: "r", 2: "n"}

    DATESTR = datetime.now().strftime("%y%m%d_%H%M%S")
    NOTE = f"Predicts NLI on CsFEVER-NLI test split. Model from LREV2022."

    return {
        "name": NAME,
        "lang": LANG,
        "lang_short": LANG_SHORT,
        "nli_root": NLI_ROOT,
        "model_name": MODEL_NAME,
        "max_length": MAX_LENGTH,
        "id2label": ID2LABEL,
        "splits": {
            # "train": Path(NLI_ROOT, "train_nli.jsonl"},
            # "dev": Path(NLI_ROOT, "dev_nli.jsonl"),
            "test": str(Path(NLI_ROOT, "test_nli.jsonl")),
        },
        "note": NOTE,
        "date": DATESTR,
    }