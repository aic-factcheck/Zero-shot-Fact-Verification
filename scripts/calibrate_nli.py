from collections import Counter, OrderedDict
from pathlib import Path
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from datasets import load_dataset, set_caching_enabled

import transformers
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
)

from aic_nlp_utils.batch import batch_apply
from aic_nlp_utils.pycfg import parse_pycfg_args, read_pycfg

from zshot_fact_verify.models.inference import split_predict

class TemperatureScaling(torch.nn.Module):
    def __init__(self):
        super(TemperatureScaling, self).__init__()
        self.T = torch.nn.Parameter(data=torch.tensor(10.0)) # single scalar parameter

    def forward(self, Z, apply_softmax=False):
        if apply_softmax:
            return F.softmax(Z/self.T, dim=1)
        else:
            return Z/self.T


class CalibrationDataset(Dataset):
    def __init__(self, X, Tcls):
        self.X = X
        self.Tcls = Tcls

    def __len__(self):
        return len(self.Tcls)

    def __getitem__(self, idx):
        return (self.X[idx], self.Tcls[idx])


def train_scaling_model(model, trainloader, epochs=50):
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    criterion = torch.nn.CrossEntropyLoss()

    for epoch in range(epochs):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            # print(inputs)
            # print(labels)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()

        print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / (i+1):.3f}\tT = {model.T.item()}')
        running_loss = 0.0
    print('Finished Training')


def main():
    args = parse_pycfg_args()

    def save_dir_fn(cfg):
        return Path(f'{cfg["model_name"]}_calibrated', f"calibrate.config.py")
    
    cfg = read_pycfg(args.pycfg, save_dir_fn=save_dir_fn)

    model_name = cfg["model_name"]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"model name: '{model_name}'")
    print(f"device: {device}")
    print(f"split =\n{cfg['split']}")

    set_caching_enabled(False)
    raw_nli = load_dataset("json", data_files={"split": cfg["split"]})

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    id2label = cfg.get("id2label", None)
    if id2label:
        label2id = {v: k for k,v in id2label.items()}
        model = AutoModelForSequenceClassification.from_pretrained(model_name, device_map="auto", id2label=id2label, label2id=label2id)
    else:
        model = AutoModelForSequenceClassification.from_pretrained(model_name, device_map="auto")

    print(f"infering logits for: '{model_name}'")
    Y, C, T = split_predict(model, tokenizer, raw_nli["split"], device=device, max_length=512, apply_softmax=False)

    scaling_model = TemperatureScaling()
    scaling_model.to(device)

    # target class names to ids
    Tid = torch.LongTensor([model.config.label2id[c] for c in T]).to(device)

    scaling_dataset = CalibrationDataset(Y, Tid)
    scaling_dataloader = DataLoader(scaling_dataset, batch_size=16)

    train_scaling_model(scaling_model, scaling_dataloader, epochs=cfg["epochs"])

    T = scaling_model.T
    print(f"T = {T}")

    last_module = list(model.children())[-1]
    last_layer = list(last_module.children())[-1]
    print(f"last layer: {last_layer}")

    with torch.no_grad():
        last_layer.weight /= T

    out_dir = str(model_name) + "_calibrated"
    print(f"saving to: {out_dir}")
    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)


if __name__ == "__main__":
    main()