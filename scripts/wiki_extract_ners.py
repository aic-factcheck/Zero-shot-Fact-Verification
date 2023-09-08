from collections import Counter, OrderedDict
import numpy as np
import pathlib
from pathlib import Path
import stanza
from tqdm import tqdm
from typing import Dict, List, Set
import ujson

import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer, BertTokenizerFast, pipeline

from aic_nlp_utils.fever import fever_detokenize
from aic_nlp_utils.json import read_jsonl, read_json, write_jsonl, write_json
from aic_nlp_utils.pycfg import parse_pycfg_args, read_pycfg

from zshot_fact_verify.wiki.load import load_corpus, create_corpus_splits, select_nei_context_for_splits

def load_ner_pipeline(model_name):
    # for general Transformer models
    ner_pipeline = pipeline("ner", model=model_name, aggregation_strategy="first", device_map='auto')
    def ner_pipeline_pairs(text):
        ner_dicts = ner_pipeline(text)
        ner_pairs = [(text[e["start"]:e["end"]], e["entity_group"]) for e in ner_dicts]
        return ner_pairs
    return ner_pipeline_pairs


def load_czert_ner_pipeline(model_name):
    # for Czech Czert models
    model = AutoModelForTokenClassification.from_pretrained(model_name)
    if torch.cuda.is_available():
        cur_dev = torch.cuda.current_device()
        print(f"torch.cuda.current_device() = {cur_dev}")
        device = torch.device(f"cuda:{cur_dev}")
    else:
        device = torch.device("cpu")

    tokenizer = BertTokenizerFast(Path(model_name, "vocab.txt"), strip_accents=False, do_lower_case=False, truncate=True, model_max_length=512)
    ner_pipeline = pipeline("ner", model=model_name, tokenizer=tokenizer, aggregation_strategy="first", device=device)
    def ner_pipeline_pairs(text):
        ner_dicts = ner_pipeline(text)
        ner_pairs = [(text[e["start"]:e["end"]], e["entity_group"]) for e in ner_dicts]
        return ner_pairs
    return ner_pipeline_pairs


def extract_ners(corpus_recs, ner_json, ner_pipeline, prefix=""):
    # for each text gives a triplet (ner, ner_type, ner-ner_type count in text)
    # the triplets are sorted by decreasing count
    ner_json_path = Path(ner_json)
    dir = ner_json_path.parent
    name = ner_json_path.name
    out_file = Path(dir, f"{prefix}{name}")
    if out_file.is_file():
        print(f'WARNING: "{str(out_file)}" exists, SKIPPING ...')
        return

    entity_dict = OrderedDict()
    key_text = f"{prefix}text"
    key_id = f"{prefix}id"
    for l in tqdm(corpus_recs):
        if key_text in l:
            text = l[key_text]
            ner_pairs = ner_pipeline(text)
            # print(f"ner_pairs = {ner_pairs}")
            ner_cnts = Counter(ner_pairs)
            ners_unique_with_counts = [(p[0], p[1], ner_cnts[(p[0], p[1])]) for p in set(ner_pairs)]
            ners_unique_with_counts = sorted(ners_unique_with_counts, key=lambda n: -n[2])
            entity_dict[l[key_id]] = ners_unique_with_counts
    ner_json_path = Path(ner_json)
    dir = ner_json_path.parent
    name = ner_json_path.name
    write_json(out_file, entity_dict, mkdir=True)


def extract_ners_stanza(corpus_recs, ner_json, lang, prefix=""):
    # for English or stanza supported languages
    # for each text gives a triplet (ner, ner_type, ner-ner_type count in text)
    # the triplets are sorted by decreasing count
    ner_json_path = Path(ner_json)
    dir = ner_json_path.parent
    name = ner_json_path.name
    out_file = Path(dir, f"{prefix}{name}")
    if out_file.is_file():
        print(f'WARNING: "{str(out_file)}" exists, SKIPPING ...')
        return

    stanza_nlp = stanza.Pipeline(lang, use_gpu = True, processors="tokenize,ner")
    entity_dict = OrderedDict()
    key_text = f"{prefix}text"
    key_id = f"{prefix}id"
    for l in tqdm(corpus_recs):
        if key_text in l:
            text = l[key_text]
            pass_doc = stanza_nlp(text)
            ner_pairs = [(ent.text, ent.type) for ent in pass_doc.ents] # text-type pairs
            ner_cnts = Counter(ner_pairs) 
            ners_unique_with_counts = [(p[0], p[1], ner_cnts[(p[0], p[1])]) for p in set(ner_pairs)]
            ners_unique_with_counts = sorted(ners_unique_with_counts, key=lambda n: -n[2])
            entity_dict[l[key_id]] = ners_unique_with_counts
    write_json(out_file, entity_dict, mkdir=True)


def main():
    args = parse_pycfg_args()

    def save_dir_fn(cfg):
        return Path(cfg["ner_root"], "ners.config.py")
    
    cfg = read_pycfg(args.pycfg, save_dir_fn=save_dir_fn)
    assert cfg["method"] in ["stanza", "transformer", "transformer_czert"], f'Unknown method: {cfg["method"]}'
    
    print(f"loading corpus '{cfg['wiki_corpus']}' ...")
    corpus, corpus_id2idx, corpus_pages = load_corpus(cfg["wiki_corpus"])

    print(f"creating splits ...", end="")
    corpus_recs_lst = create_corpus_splits(corpus, corpus_id2idx, cfg["splits"], cfg["seed"])
    print("done")

    # adds "nei{dociment number}_" prefixed context from the same document
    # NEI NERs are extracted from multiple (n_documents) documents so the final QACG datasets are better balanced
    # without this, we had too few NEIs
    corpus_recs_lst = select_nei_context_for_splits(corpus, corpus_id2idx, corpus_recs_lst, cfg["seed"], n_documents=cfg["nei_documents"])

    # for debugging
    write_json(Path(cfg["ner_root"], "corpus_recs_lst.json"), corpus_recs_lst)
    
    if cfg["method"] == "transformer_czert":
        ner_pipeline = load_czert_ner_pipeline(cfg["model_name"])
    elif cfg["method"] == "transformer":
        ner_pipeline = load_ner_pipeline(cfg["model_name"])

    nei_prefixes = [f"nei{i+1}_" for i in range(cfg["nei_documents"])]
    for corpus_recs, split in zip(corpus_recs_lst, cfg["splits"]):
        for prefix in [""] + nei_prefixes:
        # for prefix in ["nei_"]:
            print(f"\n--------------- split: {prefix}{split['name']} -----------------")
            if cfg["method"] == "stanza":
                extract_ners_stanza(corpus_recs, split["file"], cfg["lang_short"], prefix=prefix)
            else:
                extract_ners(corpus_recs, split["file"], ner_pipeline, prefix=prefix)

        # combine NEI exports to single file
        dir = Path(split["file"]).parent
        name = Path(split["file"]).name
        entity_dict = {}
        n_total = 0
        n_duplicates = 0
        for nei_prefix in nei_prefixes:
            nei_file = Path(dir, f"{nei_prefix}{name}")
            entity_dict_part = read_json(nei_file)
            print(f"merging '{str(nei_file)}'")
            for k, v in entity_dict_part.items():
                if k in entity_dict:
                    print(f'WARNING: "{k}" already in extracted entities!')
                    n_duplicates += 1
                n_total += 1
                entity_dict[k] = v
        print(f"{n_duplicates} duplicate documents in {n_total}")
        write_json(Path(dir, f"nei_{name}"), entity_dict, mkdir=True)

        # delete original files
        for nei_prefix in nei_prefixes:
            Path(dir, f"{nei_prefix}{name}").unlink()


if __name__ == "__main__":
    main()