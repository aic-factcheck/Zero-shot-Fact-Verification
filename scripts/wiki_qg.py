from pathlib import Path
from pprint import pprint

from aic_nlp_utils.pycfg import parse_pycfg_args, read_pycfg
from aic_nlp_utils.json import read_jsonl, read_json, write_json, write_jsonl

from zshot_fact_verify.models.arguments import ModelArguments, DataTrainingArguments
from zshot_fact_verify.models.load import load_tokenizer_and_model, find_last_checkpoint
from zshot_fact_verify.qg.question_generation import BatchQuestionGenerator, generate_questions
from zshot_fact_verify.wiki.load import load_corpus, create_corpus_splits, select_nei_context_for_splits, load_nei_ners

def main ():
    args = parse_pycfg_args()

    def save_dir_fn(cfg):
        return Path(cfg["qg_root"], "qgs.config.py")
    
    cfg = read_pycfg(args.pycfg, save_dir_fn=save_dir_fn)

    qg_modes = [m.lower() for m in cfg["qg_modes"]]

    for m in qg_modes:
        assert m in ["regular", "nei"], f"Unknown mode: {m}"
    
    corpus, corpus_id2idx, corpus_pages = load_corpus(cfg["wiki_corpus"])
    corpus_recs_lst = create_corpus_splits(corpus, corpus_id2idx, cfg["splits"], cfg["seed"])
    corpus_recs_lst = select_nei_context_for_splits(corpus, corpus_id2idx, corpus_recs_lst, cfg["seed"], n_documents=cfg["nei_documents"])

    # for debugging
    # write_json(Path(cfg["qg_root"], "corpus_recs_lst.json"), corpus_recs_lst)

    model_args = ModelArguments(model_name_or_path=cfg["model_name"])
    tokenizer, model, data_collator = load_tokenizer_and_model(model_args, lang=cfg["lang"], fp16=True)

    batch_question_generator = BatchQuestionGenerator(tokenizer, model, highlight=cfg["highlight"], padding=True, debug=False)

    for corpus_recs, split in zip(corpus_recs_lst, cfg["splits"]):
        name = split['name']
        print(f"--------------- split: {name} size = {len(corpus_recs)}-----------------")

        ner_file = Path(cfg["ner_root"], f"{name}_ners.json")
        original_ners = read_json(ner_file)
        print(f"loaded {len(original_ners)} passages' NERs from {ner_file}")

        # for debugging
        # write_json(Path(cfg["qg_root"], f"{name}_ners_extracted.json"), original_ners)

        if "regular" in qg_modes:
            print(f"--------------- regular mode questions -----------------")

            qg_file = Path(cfg["qg_root"], f"{name}_qgs.json")
            if qg_file.exists():
                print(f"{qg_file} exists SKIPPING...")
            else:
                generate_questions(corpus_recs, original_ners, qg_file, batch_question_generator)
        
        if "nei" in qg_modes:
            print(f"--------------- NEI mode questions -----------------")

            nei_ner_file = Path(cfg["ner_root"], f"nei_{name}_ners.json")
            # print(f"DEBUG: nei_ner_file = {nei_ner_file}")
            nei_ners = load_nei_ners(corpus_recs, original_ners, nei_ner_file, n_documents=cfg["nei_documents"])
            print(f"loaded {len(nei_ners)} passages' NEI NERs from {nei_ner_file}")

            # for debugging
            # write_json(Path(cfg["qg_root"], f"nei_{name}_ners_extracted.json"), nei_ners)

            qg_file = Path(cfg["qg_root"], f"nei_{name}_qgs.json")
            if qg_file.exists():
                print(f"{qg_file} exists SKIPPING...")
            else:
                generate_questions(corpus_recs, nei_ners, qg_file, batch_question_generator, nei=True, n_documents=cfg["nei_documents"])

if __name__ == "__main__":
    main()