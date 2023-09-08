from collections import OrderedDict
from pathlib import Path
from typing import Dict, List, Set, Union

import numpy as np
from tqdm import tqdm
import torch

from aic_nlp_utils.batch import batch_apply
from aic_nlp_utils.json import read_jsonl, read_json, write_json, write_jsonl

from zshot_fact_verify.models.arguments import ModelArguments
from zshot_fact_verify.models.load import load_tokenizer_and_model

class SameDocumentNERReplacementGenerator:
    # same interface as Distractor_Generation by the original authors

    def __init__(self, seed=1234):
        self.rng = np.random.RandomState(seed)

    def get_options(self, answer, entity, passage_entities, **kwargs):
        ent_name, ent_type = entity.split(":::")
        selected_entity_names = set()
        for passage_entity in passage_entities:
            pent_name, pent_type = passage_entity.split(":::")
            if pent_type == ent_type and pent_name != ent_name:
                selected_entity_names.add(pent_name)
        if len(selected_entity_names) == 0:
            return None
        selected_entity_names = list(selected_entity_names)
        selected_entity_name = self.rng.choice(selected_entity_names)
        selected_entity = (selected_entity_name, ent_type)
        # print(f"{entity} -> {selected_entity}")
        return selected_entity
    

class ClaimGenerator:
    def __init__(self, replacement_generator, corpus_recs, QA2D_model_path, lang, device="cuda"):
        # QA2D model object
        print('Loading QA2D module >>>>>>>>')
        model_args = ModelArguments(model_name_or_path=QA2D_model_path)
        self.tokenizer, self.model, data_collator = load_tokenizer_and_model(model_args, lang=lang)
        print(f'Running on device: {device}')
        # self.model, self.tokenizer = model, tokenizer # TODO REMOVE
        self.device = device
        self.model.to(device)

        self.replacement_generator = replacement_generator

        self.corpus_recs = corpus_recs

    def predict(self, inputs, max_source_length=1024, batch_size=16):
        def pred_func(input_texts: List[str]) -> List[str]:
            with torch.no_grad():
                X = self.tokenizer(input_texts, max_length=max_source_length, padding=True, truncation=True, return_tensors="pt")
                X = {k: X[k].to(self.device) for k in X.keys()}
                Y = self.model.generate(**X, max_new_tokens=768)
                output_texts = self.tokenizer.batch_decode(
                    Y, skip_special_tokens=True, clean_up_tokenization_spaces=True
                )
            return output_texts
            
        predictions = batch_apply(pred_func, inputs, batch_size=batch_size)
        return predictions

    def _load_passage_entities(self, id_, ners):
        if id_ not in ners: # may happen for NEI
            return []
        passage_entities = []
        for ent_text, ent_type, _ in ners[id_]:
            passage_entities.append(f'{ent_text}:::{ent_type}') # group by entity name and type as in the QAS file
        return passage_entities
    
    def _load_precomputed_qas_for_entities(self, id_, passage_entities, qgs):
        if id_ not in qgs:
            print(f"missing id: {id_}")
            return None
        QA_for_sample = qgs[id_]
        QA_pairs = []
        for entity in passage_entities:
            if entity in QA_for_sample:
                ent_text, ent_type = entity.split(':::')
                question, answer = QA_for_sample[entity]
                QA_pairs.append({'question': question, 'answer': answer, 'answer_type': ent_type})
            else:
                print(f"missing entity: {entity} for id: {id_}")
                return None
        if len(QA_pairs) == 0:
            print(f"zero length pairs for id: {id_}")
            return None
        return QA_pairs 
        

    def generate_supported_claims(self, id_, ners, qgs):
        # Step 1: load entities in text
        passage_entities = self._load_passage_entities(id_, ners)
        if len(passage_entities) == 0: # no NERs
            return None 

        # Step 2: load precomputed QAs for entities
        QA_pairs = self._load_precomputed_qas_for_entities(id_, passage_entities, qgs)
        if QA_pairs is None:
            return None

        # Step 3: QA2D
        # to_predict = [qa['question'] + ' [SEP] ' + qa['answer'] for qa in QA_pairs] # original model
        to_predict = [qa['answer'] + '</s>' + qa['question'] for qa in QA_pairs]
        results = []
        # try:
        results = self.predict(to_predict)
        # except:
            # return None
        if len(results) == 0:
            print(f"zero length results for id: {id_}")
            return None

        assert len(results) == len(QA_pairs)

        claims_for_sample = OrderedDict()
        for ent, claim in zip(passage_entities, results):
            claims_for_sample[ent] = claim
        return claims_for_sample
    

    def generate_refute_local_claims(self, id_, ners, qgs):
        # Step 1: load entities in text
        passage_entities = self._load_passage_entities(id_, ners)
        if len(passage_entities) == 0: # no NERs
            return None 
        
        # Step 2: get entity replacement
        entity_replacement_dict = {} # get replacement beforehand to save time

        valid_entities = set()
        for ent in passage_entities:
            ent_text, _ = ent.split(':::')
            replacement = self.replacement_generator.get_options(ent_text, entity=ent, passage_entities=passage_entities)
            if replacement is not None:
                entity_replacement_dict[ent_text] = replacement
                valid_entities.add(ent)
        # print(f"entity_replacement_dict={entity_replacement_dict}")

        # Step 3: load precomputed QAs for entities
        QA_pairs = self._load_precomputed_qas_for_entities(id_, passage_entities, qgs)
        if QA_pairs is None:
            return None

        # Step 4: Answer Replacement
        to_predict = []
        replace_type = []
        replace_keys = []
    
        for qa in QA_pairs:
            ans_ent_text = qa['answer']
            ans_ent_type = qa['answer_type']
            if ans_ent_text == "" or ans_ent_type == "":
                continue
            replacement = entity_replacement_dict.get(ans_ent_text)
            if replacement is not None:
                # print(f'"{ans_ent_text}:::{ans_ent_type}" -> "{replacement}"')
                # predict_input = qa['question'] + ' [SEP] ' + replacement[0] # original model
                # predict_input = qa['question'] + '</s>' + replacement[0] # ERROR
                predict_input = replacement[0] + '</s>' + qa['question']
                # print(f">>> {predict_input}")
                to_predict.append(predict_input)
                replace_keys.append(f"{ans_ent_text}:::{ans_ent_type}")
                replace_type.append(ans_ent_type)

        # Step 5: QA2D
        if len(to_predict) == 0:
            return None
        # results = []
        # try:
        results = self.predict(to_predict)
            # print(f"results={results}")
        # except:
            # return None
        if len(results) == 0:
            return None
        
        claims_for_sample = OrderedDict()
        for ent, claim in zip(replace_keys, results):
            claims_for_sample[ent] = claim
        return claims_for_sample


    def generate(self, ners, qgs, claims_json, claim_type: str, save_every=0, cont=False):
        claim_type = claim_type.lower()
        assert claim_type in ["support", "refute", "nei"]
        start = 0
        if Path(claims_json).is_file():
            if cont:
                generated_claims = read_json(claims_json)
                print(f"file exists: {claims_json}, completed: {len(generated_claims)-1}/{len(self.corpus_recs)}")
                start = len(generated_claims)
            else:
                # print("--------------FIX!!!!!!!!!!!-------------------------")
                # generated_claims = read_json(claims_json)
                raise FileExistsError(f"File already exists: {claims_json} !!!")
        else:
            generated_claims = dict() # ordered since P3.7
        cnt = 1
        for sample in tqdm(self.corpus_recs[start:], initial=start, total=len(self.corpus_recs)):
            id_ = str(sample['id'])

            if claim_type == "support":
                claims = self.generate_supported_claims(id_, ners, qgs)
            elif claim_type == "refute":
                claims = self.generate_refute_local_claims(id_, ners, qgs)
            elif claim_type == "nei":
                claims = self.generate_supported_claims(id_, ners, qgs)
            if claims is None:
                claims = {}
            generated_claims[id_] = claims
            cnt += 1
            if save_every > 0 and cnt % save_every == 0:
                write_json(claims_json, generated_claims, mkdir=True)

        write_json(claims_json, generated_claims, mkdir=True)