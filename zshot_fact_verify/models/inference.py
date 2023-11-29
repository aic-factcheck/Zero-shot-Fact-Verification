import numpy as np
import torch
import torch.nn.functional as F

from aic_nlp_utils.batch import batch_apply

def split_predict(model, tokenizer, split, batch_size=128, device="cuda", max_length=128, 
                  apply_softmax=False, one_hot_targets=False, to_numpy=False):
    def predict(inputs):
        X = tokenizer(inputs, max_length=max_length, padding=True, truncation=True, return_tensors="pt")
        input_ids = X["input_ids"].to(device)
        attention_mask = X["attention_mask"].to(device)
        with torch.no_grad():
            Y = model(input_ids=input_ids, attention_mask=attention_mask).logits
            return Y
        
    inputs = [[claim, context] for claim, context in zip(split["claim"],  split["context"])]
    # inputs = [[context, claim] for claim, context in zip(split["claim"],  split["context"])] # SWITCHED CTX and CLAIM!!!
    Ys = batch_apply(predict, inputs, batch_size=batch_size, show_progress=True)
    Y = torch.vstack(Ys)

    if apply_softmax:
        Y = F.softmax(Y, dim=1)

    C = [model.config.id2label[id_.item()] for id_ in Y.argmax(dim=1)]
    if one_hot_targets:
        if to_numpy:
            T = np.array([model.config.label2id[l] for l in split["label"]])
        else:
            T = torch.IntTensor([model.config.label2id[l] for l in split["label"]]) 
    else:
        T = [l for l in split["label"]]
        
    if to_numpy:
        Y = Y.detach().cpu().numpy()
    return Y, C, T