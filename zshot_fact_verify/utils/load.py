from collections import defaultdict
import numpy as np
from pathlib import PosixPath
from typing import Dict, List, Set
from aic_nlp_utils.json import read_jsonl, read_json, write_jsonl, write_json

def extract_random_evidence_pages(corpus_id2idx: Dict, corpus, sizes: List[int], exclude_ids: set, seed=1234):
    N = np.sum(sizes)
    rng = np.random.RandomState(seed)
    all_ids = set(corpus_id2idx.keys())
    all_ids = sorted(list(all_ids.difference(exclude_ids))) # sort needed for determinism
    selected_ids = rng.choice(all_ids, N, replace=False)
    # split_ids = []
    offset = 0
    corpus_records = []
    for n in sizes:
        ids = list(selected_ids[offset:offset+n])
        # split_ids.append(ids)
        corpus_records.append([corpus[corpus_id2idx[id_]] for id_ in ids])
        offset += n
    # return split_ids, corpus_records
    return corpus_records

def load_corpus(corpus):
    corpus = read_jsonl(corpus)
    assert len(corpus) > 0, "empty corpus!"
    if "did" not in corpus[0]:
        for rec in corpus:
            assert "_" in rec["id"], "Expecting id format did_bid!"
            rec["did"] = rec["id"].split("_")[0]
            rec["bid"] = int(rec["id"].split("_")[1])

    corpus_id2idx = {r["id"]: i for i, r in enumerate(corpus)}
    corpus_pages = set(r["did"] for r in corpus)
    
    print(f"imported {len(corpus_pages)} corpus pages.")
    return corpus, corpus_id2idx, corpus_pages

def create_corpus_splits(corpus, corpus_id2idx, splits, seed):
    split_sizes = [s["size"] for s in splits]
    corpus_recs_lst  = extract_random_evidence_pages(
        corpus_id2idx, 
        corpus, 
        sizes=split_sizes, 
        exclude_ids=set(),
        seed=seed)
    return corpus_recs_lst


def select_nei_context_for_splits(corpus, corpus_id2idx, corpus_recs_lst, seed, n_documents=1):
    did2ids = defaultdict(list)
    for sample in corpus:
        did2ids[sample["did"]].append(sample["id"])

    rng = np.random.RandomState(seed)
    for corpus_recs in corpus_recs_lst:
        for sample in corpus_recs:
            nei_context_ids = did2ids[sample["did"]].copy()
            nei_context_ids.remove(sample["id"])
            for i in range(n_documents):
                if len(nei_context_ids) > 0:
                    nei_context_id = rng.choice(nei_context_ids)
                    nei_context_ids.remove(nei_context_id)
                    nei_context_idx = corpus_id2idx[nei_context_id]
                    nei_context = corpus[nei_context_idx]
                    sample[f"nei{i+1}_id"] = nei_context["id"]
                    sample[f"nei{i+1}_bid"] = nei_context["bid"]
                    sample[f"nei{i+1}_text"] = nei_context["text"]
    return corpus_recs_lst


def load_nei_ners(corpus_recs, original_ners, nei_ner_json, translate_ids=False, n_documents=1):
    # loads NEI NERs, but also removes those extracted from the the original context
    # if translate_ids == True the imported NEI context ids are changed to original context ids
    nei_ners_all = read_json(nei_ner_json)
    ret = {}
    missing_nei_records = 0
    removed_ners_pct = []
    for sample in corpus_recs:
        id_ = sample["id"]
        
        # available NERs to prevent NEI duplicities
        ner_set = set([n[0] for n in original_ners[id_]])

        for i in range(n_documents):
            if f"nei{i+1}_id" not in sample:
                missing_nei_records += 1
                # print(f"missing 'nei{i+1}_id' in {sample.keys()}")
                continue
            nei_id = sample[f"nei{i+1}_id"]
            nei_ners = [n for n in nei_ners_all[nei_id] if n[0] not in ner_set]
            
            # prevent duplicities for multiple NEI documents
            for n in nei_ners:
                ner_set.add(n[0])

            n_before = len(nei_ners_all[nei_id])
            n_after = len(nei_ners)
            if n_before == 0:
                removed_ners_pct.append(0.0)
            else:
                removed_ners_pct.append(100*(n_before-n_after)/n_before)

            if translate_ids:
                ret[id_] = nei_ners
            else:
                ret[nei_id] = nei_ners

    print(f"load_nei_ners: {missing_nei_records} missing NEI records (corpus len={len(corpus_recs)}), average # removed: {np.mean(removed_ners_pct)}%")
    return ret