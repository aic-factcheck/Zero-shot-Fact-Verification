from collections import Counter, OrderedDict
from pathlib import Path
from tqdm import tqdm

import torch

from aic_nlp_utils.json import read_jsonl, read_json, write_json, write_jsonl
from aic_nlp_utils.pycfg import parse_pycfg_args, read_pycfg

from zshot_fact_verify.models.load import load_tokenizer_and_model, find_last_checkpoint
from zshot_fact_verify.utils.load import load_corpus, create_corpus_splits, select_nei_context_for_splits, load_nei_ners
from zshot_fact_verify.qa2d.qa2d import SameDocumentNERReplacementGenerator, ClaimGenerator

def main ():
    args = parse_pycfg_args()

    def save_dir_fn(cfg):
        return Path(cfg["claim_root"], "qa2d.config.py")
    
    cfg = read_pycfg(args.pycfg, save_dir_fn=save_dir_fn)

    claim_types = sorted(list(set([c.lower() for c in cfg["claim_types"]])))
    for c in claim_types:
        assert c in set(["support", "refute", "nei"]), f"Unknown claim type: {c}"
    
    corpus, corpus_id2idx, corpus_pages = load_corpus(cfg["corpus"])
    corpus_recs_lst = create_corpus_splits(corpus, corpus_id2idx, cfg["splits"], cfg["seed"])
    corpus_recs_lst = select_nei_context_for_splits(corpus, corpus_id2idx, corpus_recs_lst, cfg["seed"], n_documents=cfg["nei_documents"])

    device = "cuda" if torch.cuda.is_available() else "cpu"

    replacement_generator = SameDocumentNERReplacementGenerator()

    for corpus_recs, split in zip(corpus_recs_lst, cfg["splits"]):
        name = split['name']
        print(f"--------------- split: {name} -----------------")


        ner_file = Path(cfg["ner_root"], f"{name}_ners.json")
        original_ners = read_json(ner_file)
        print(f"loaded {len(original_ners)} passages' NERs from {ner_file}")
        
        qgs_file = Path(cfg["qg_root"], f"{name}_qgs.json")
        original_qgs = read_json(qgs_file)
        print(f"loaded {len(original_qgs)} passages' questions from {qgs_file}")

        if "nei" in claim_types:
            nei_ner_file = Path(cfg["ner_root"], f"nei_{name}_ners.json")
            nei_ners = load_nei_ners(corpus_recs, original_ners, nei_ner_file, translate_ids=True, n_documents=cfg["nei_documents"])
            print(f"loaded {len(nei_ners)} passages' NEI NERs from {nei_ner_file}")
            
            nei_qgs_file = Path(cfg["qg_root"], f"nei_{name}_qgs.json")
            nei_qgs = read_json(nei_qgs_file)
            print(f"loaded {len(nei_qgs)} passages' NEI questions from {nei_qgs_file}")

        claim_generator = ClaimGenerator(replacement_generator, 
                                    corpus_recs, 
                                    QA2D_model_path=cfg["model_name"], 
                                    lang=cfg["lang"],
                                    device=device)

        for c in claim_types:
            print(f"--------------- claim type: {c} -----------------")
            ners_ = nei_ners if c == "nei" else original_ners
            qgs_ = nei_qgs if c == "nei" else original_qgs

            claim_generator.generate(ners_, qgs_, 
                                     claims_json=Path(cfg["claim_root"], f"{name}_{c}.json"), 
                                     claim_type=c, save_every=100, cont=True)

if __name__ == "__main__":
    main()