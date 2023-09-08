from pathlib import Path
from aic_nlp_utils.json import read_jsonl, read_json, write_json, write_jsonl


def main():
    # fixes errors in Slovak ner extraction when multiple paragraphs are merged

    LANG = "sk_SK"
    LANG_SHORT = "sk"
    METHOD = "transformer"
    MODEL_NAME = "/mnt/data/factcheck/models/slovakbert-ner"
    MODEL_SHORT = "crabz_slovakbert-ner"
    NER_DIR = MODEL_SHORT

    # DATE = "20230220"
    DATE = "20230801"
    WIKI_ROOT = f"/mnt/data/factcheck/wiki/{LANG_SHORT}/{DATE}"
    QACG_ROOT = f"{WIKI_ROOT}/qacg"
    NER_ROOT = Path(QACG_ROOT, "ner", NER_DIR)
    
    print(f"NER_ROOT: {NER_ROOT}")

    for fname in NER_ROOT.glob("*.json"):
        print(f"processing: {fname}")
        recs = read_json(fname)
        rets = {}
        for bid, ners in recs.items():
            for ner in ners:
                nertxt = ner[0]
                if "\n\n" in nertxt:
                    candidates = nertxt.split("\n\n")
                    candidates = [c.strip(":;.?!,\"'()[]\{\}") for c in candidates]
                    candidates2 = [c for c in candidates if len(c) > 1 and (c[0] in '0123456789' or c[0].upper() == c[0])]
                    candidate = candidates2[0] if len(candidates2) > 0 else candidates[0]
                    ner[0] = candidate
                else:
                    ner[0] = nertxt.strip(":;.?!,\"'()[]\{\}")
            rets[bid] = ners
        Path(fname).rename(f"{fname}.orig")
        write_json(fname, rets)

main()