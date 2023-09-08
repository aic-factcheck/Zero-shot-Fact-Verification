from collections import defaultdict, Counter, OrderedDict
from pathlib import Path
from tqdm import tqdm

import torch
import textwrap

from aic_nlp_utils.json import read_jsonl, read_json, write_json, write_jsonl

class BatchQuestionGenerator:
    def __init__(self, tokenizer, model, highlight=False, highlight_tag="<hl>", max_source_length=1024, padding=False, device="cuda", debug=False):
        self.tokenizer = tokenizer
        self.model = model.to(device)
        self.highlight = highlight
        self.highlight_tag = highlight_tag
        self.max_source_length = max_source_length
        self.padding = padding
        self.device = device
        self.debug = debug

    def generate(self, contexts, answers, batch_size=32):
        def highlight_fun(answer, context):
            offset = context.index(answer)
            return f"{context[:offset]}<hl>{answer}<hl>{context[offset + len(answer):]}"

        n = len(contexts)
        assert n == len(answers), (n, len(answers))
        offset = 0
        failures = 0
        predictions = []
        while offset < n:
            last = min(offset + batch_size, n)
            if self.highlight:
                inputs = []
                for context, answer in zip(contexts[offset:last], answers[offset:last]):
                    # if answer in context:
                    inputs.append(highlight_fun(answer, context) )
                    # else:
                        # failures += 1
            else:
                inputs = [answer + "</s>" + context for context, answer in zip(contexts[offset:last], answers[offset:last])]
            model_inputs = self.tokenizer(inputs, max_length=self.max_source_length, padding=self.padding, truncation=True, return_tensors="pt")
            model_inputs = {k: v.to(self.device) for k, v in model_inputs.items()}
            with torch.no_grad():
                Y = self.model.generate(**model_inputs, max_new_tokens=768)
                batch_predictions = self.tokenizer.batch_decode(
                    Y, skip_special_tokens=True, clean_up_tokenization_spaces=True
                )
            predictions += batch_predictions
            offset += batch_size

        assert n == len(predictions)
        if self.debug:
            for input, pred in zip(inputs, predictions):
                print(textwrap.fill(input))
                print()
                print(pred)
                print("----------------------------")
        # print(f"#failures: {failures}, #predictions: {len(predictions)}/{n}")
        return predictions


def generate_questions(corpus_recs, ners, qgs_json, generator, nei=False, n_documents=1):
    # QG NLP object
    qgs = defaultdict(dict)
    invalid_sample = 0
    for l in tqdm(corpus_recs):

        for i in range(n_documents):

            if nei:
                if f'nei{i+1}_id' not in l: # no NEI context available (should be rare)
                    continue
                id_ = str(l[f'nei{i+1}_id'])
            else:
                assert n_documents == 1, n_documents
                id_ = str(l['id'])

            if id_ not in ners: # no NERs in this text
                continue
            entities = ners[id_]

            # print(f"QG: {l}")

            # create a batch
            contexts, answers = [], []
            for ent_text, ent_type, ent_cnt in entities:
                if nei:
                    # concatenate original and NEI contexts, order them based on full original document (block ids)
                    org_ctx, org_bid = l['text'], int(l['bid'])
                    nei_ctx, nei_bid = l[f'nei{i+1}_text'], int(l[f'nei{i+1}_bid'])
                    
                    assert org_bid != nei_bid, (org_bid, nei_bid)
                    
                    if org_bid < nei_bid:
                        ctx = org_ctx + "\n\n" + nei_ctx
                    else:
                        ctx = nei_ctx + "\n\n" + org_ctx
                else:
                    ctx = l['text']
                # print("    ----------------------------")
                # print(f"   CTX: {ctx}")
                contexts.append(ctx)
                answers.append(ent_text)

            # question generation
            if len(contexts) > 0 and len(contexts) == len(answers):
                questions = []
                # try:
                questions = generator.generate(contexts, answers)
                # questions = [a + "?" for a in answers] # debug
                # except:
                    # invalid_sample += 1

                if len(questions) == 0:
                    continue
                
                assert len(questions) == len(contexts)

                # save results
                for entity, question, answer, context in zip(entities, questions, answers, contexts):
                    ent_text, ent_type, _ = entity
                    qgs[str(l['id'])][f'{ent_text}:::{ent_type}'] = [question, answer]
            else:
                invalid_sample += 1

    # print(f'#invalid samples: {invalid_sample}')
    if qgs_json:
        Path(qgs_json).parent.mkdir(parents=True, exist_ok=True)
        write_json(qgs_json, qgs)
    return qgs
