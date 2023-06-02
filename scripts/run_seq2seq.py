#!/usr/bin/env python
# coding=utf-8
# Copyright 2021 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning the library models for sequence to sequence.

Based on Transformers 4.26: examples/pytorch/summarization/run_summarization
TODO:
  - move it to aic-nlp-utils
Changes:
  - split into multiple py files
  - renamed text_column -> input_columns: multiple columns separated by ":" can be given
  - renamed summary_column -> target_columns: multiple columns separated by ":" can be given
  - added column_separator = " ": separator for multiple column concatenation
  - TODO: correct JSONL extension
"""
# You can also adapt this script on your own sequence to sequence task. Pointers for this are left as comments.

import logging
import os
import sys

import datasets
import numpy as np
from datasets import load_dataset

import transformers
from transformers import (
    HfArgumentParser,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    set_seed,
)
from transformers.utils import check_min_version, send_example_telemetry
from transformers.utils.versions import require_version

from arguments import ModelArguments, DataTrainingArguments
from load import load_tokenizer_and_model, find_last_checkpoint
from metrics import ROUGEMetric

# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.26.0")

require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/summarization/requirements.txt")

logger = logging.getLogger(__name__)

summarization_name_mapping = {
    "amazon_reviews_multi": ("review_body", "review_title"),
    "big_patent": ("description", "abstract"),
    "cnn_dailymail": ("article", "highlights"),
    "orange_sum": ("text", "summary"),
    "pn_summary": ("article", "summary"),
    "psc": ("extract_text", "summary_text"),
    "samsum": ("dialogue", "summary"),
    "thaisum": ("body", "summary"),
    "xglue": ("news_body", "news_title"),
    "xsum": ("document", "summary"),
    "wiki_summary": ("article", "highlights"),
    "multi_news": ("document", "summary"),
}

def get_raw_datasets(model_args, data_args):
    # Get the datasets: you can either provide your own CSV/JSON training and evaluation files (see below)
    # or just provide the name of one of the public datasets available on the hub at https://huggingface.co/datasets/
    # (the dataset will be downloaded automatically from the datasets Hub).
    #
    # For CSV/JSON files this script will use the first column for the input texts and the second column for the
    # targets (unless you specify column names for this with the `input_columns` and `target_columns` arguments). 
    # Multiple columns can be concatenated using column_separator.
    #
    # In distributed training, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.
    if data_args.dataset_name is not None:
        # Downloading and loading a dataset from the hub.
        raw_datasets = load_dataset(
            data_args.dataset_name,
            data_args.dataset_config_name,
            cache_dir=model_args.cache_dir,
            use_auth_token=True if model_args.use_auth_token else None,
        )
    else:
        data_files = {}
        if data_args.train_file is not None:
            data_files["train"] = data_args.train_file
            extension = data_args.train_file.split(".")[-1]
        if data_args.validation_file is not None:
            data_files["validation"] = data_args.validation_file
            extension = data_args.validation_file.split(".")[-1]
        if data_args.test_file is not None:
            data_files["test"] = data_args.test_file
            extension = data_args.test_file.split(".")[-1]
        raw_datasets = load_dataset(
            extension,
            data_files=data_files,
            cache_dir=model_args.cache_dir,
            use_auth_token=True if model_args.use_auth_token else None,
        )
    # See more about loading any type of standard or custom dataset (from files, python dict, pandas DataFrame, etc) at
    # https://huggingface.co/docs/datasets/loading_datasets.html.
    return raw_datasets


def get_columns(raw_datasets, training_args, data_args):
   # Preprocessing the datasets.
    # We need to tokenize inputs and targets.
    if training_args.do_train:
        column_names = raw_datasets["train"].column_names
    elif training_args.do_eval:
        column_names = raw_datasets["validation"].column_names
    elif training_args.do_predict:
        column_names = raw_datasets["test"].column_names
    else:
        logger.info("There is nothing to do. Please pass `do_train`, `do_eval` and/or `do_predict`.")
        return

    # Get the column names for input/target.
    dataset_columns = summarization_name_mapping.get(data_args.dataset_name, None)
    if data_args.input_columns is None:
        input_columns = [dataset_columns[0]] if dataset_columns is not None else [column_names[0]]
    else:
        input_columns = data_args.input_columns.split(":")
        for input_column in input_columns:
            if input_column not in column_names:
                raise ValueError(
                    f"--input_columns' value '{data_args.input_columns}' needs to be a list of: {', '.join(column_names)} separated by :"
                )
    if data_args.target_columns is None:
        target_columns = [dataset_columns[1]] if dataset_columns is not None else [column_names[1]]
    else:
        target_columns = data_args.target_columns.split(":")
        for target_column in target_columns:
            if target_column not in column_names:
                raise ValueError(
                    f"--target_columns' value '{data_args.target_columns}' needs to be a list of: {', '.join(column_names)} separated by :"
                )
    return input_columns, target_columns, column_names


def prepare_train_dataset(raw_datasets, data_args, training_args, preprocess_function, column_names):
    if "train" not in raw_datasets:
        raise ValueError("--do_train requires a train dataset")
    train_dataset = raw_datasets["train"]
    if data_args.max_train_samples is not None:
        max_train_samples = min(len(train_dataset), data_args.max_train_samples)
        train_dataset = train_dataset.select(range(max_train_samples))
    with training_args.main_process_first(desc="train dataset map pre-processing"):
        train_dataset = train_dataset.map(
            preprocess_function,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            remove_columns=column_names,
            load_from_cache_file=not data_args.overwrite_cache,
            desc="Running tokenizer on train dataset",
        )
    return train_dataset


def prepare_eval_dataset(raw_datasets, data_args, training_args, preprocess_function, column_names):
    max_target_length = data_args.val_max_target_length
    if "validation" not in raw_datasets:
        raise ValueError("--do_eval requires a validation dataset")
    eval_dataset = raw_datasets["validation"]
    if data_args.max_eval_samples is not None:
        max_eval_samples = min(len(eval_dataset), data_args.max_eval_samples)
        eval_dataset = eval_dataset.select(range(max_eval_samples))
    with training_args.main_process_first(desc="validation dataset map pre-processing"):
        eval_dataset = eval_dataset.map(
            preprocess_function,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            remove_columns=column_names,
            load_from_cache_file=not data_args.overwrite_cache,
            desc="Running tokenizer on validation dataset",
        )
    return eval_dataset, max_target_length


def prepare_predict_dataset(raw_datasets, data_args, training_args, preprocess_function, column_names):
    max_target_length = data_args.val_max_target_length
    if "test" not in raw_datasets:
        raise ValueError("--do_predict requires a test dataset")
    predict_dataset = raw_datasets["test"]
    if data_args.max_predict_samples is not None:
        max_predict_samples = min(len(predict_dataset), data_args.max_predict_samples)
        predict_dataset = predict_dataset.select(range(max_predict_samples))
    with training_args.main_process_first(desc="prediction dataset map pre-processing"):
        predict_dataset = predict_dataset.map(
            preprocess_function,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            remove_columns=column_names,
            load_from_cache_file=not data_args.overwrite_cache,
            desc="Running tokenizer on prediction dataset",
        )
    return predict_dataset, max_target_length


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, Seq2SeqTrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Sending telemetry. Tracking the example usage helps us better allocate resources to maintain them. The
    # information sent is the one passed as arguments along with your Python/PyTorch versions.
    send_example_telemetry("run_seq2seq", model_args, data_args)

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(logging.WARNING)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    last_checkpoint = find_last_checkpoint(training_args)

    # Set seed before initializing model.
    set_seed(training_args.seed)

    raw_datasets = get_raw_datasets(model_args, data_args)
    tokenizer, model, data_collator = load_tokenizer_and_model(model_args, lang=data_args.lang, fp16=training_args.fp16)
    input_columns, target_columns, column_names = get_columns(raw_datasets, training_args, data_args)


    if training_args.label_smoothing_factor > 0 and not hasattr(model, "prepare_decoder_input_ids_from_labels"):
        logger.warning(
            "label_smoothing is enabled but the `prepare_decoder_input_ids_from_labels` method is not defined for"
            f"`{model.__class__.__name__}`. This will lead to loss being calculated twice and will take up more memory"
        )

    prefix = data_args.source_prefix if data_args.source_prefix is not None else ""

    # Temporarily set max_target_length for training.
    max_target_length = data_args.max_target_length
    padding = "max_length" if data_args.pad_to_max_length else False

    def preprocess_function(examples):
        # remove pairs where at least one record is None

        def concat_columns(examples, idx, columns):
            res = examples[columns[0]][idx]
            if res is None:
                return None
            for col in columns[1:]:
                ctext = examples[col][idx]
                if ctext is None:
                    return None
                res += data_args.column_separator + ctext
            return res
        
        def highlight_columns(examples, idx, columns):
            # surround the first appearance of columns[0] in columns[1] using highlight_tag
            assert len(columns) == 2, len(columns)
            target = examples[columns[0]][idx]
            context = examples[columns[1]][idx]
            offset = context.index(target)
            res = f"{context[:offset]}<hl>{target}<hl>{context[offset + len(target):]}"
            return res

        inputs, targets = [], []
        column_f = highlight_columns if data_args.highlight else concat_columns
        for i in range(len(examples[input_columns[0]])):
            input = column_f(examples, i, input_columns)
            target = concat_columns(examples, i, target_columns)
            if input and target:
                inputs.append(input)
                targets.append(target)

        inputs = [prefix + inp for inp in inputs]
        model_inputs = tokenizer(inputs, max_length=data_args.max_source_length, padding=padding, truncation=True)

        # Tokenize targets with the `text_target` keyword argument
        labels = tokenizer(text_target=targets, max_length=max_target_length, padding=padding, truncation=True)

        # If we are padding here, replace all tokenizer.pad_token_id in the labels by -100 when we want to ignore
        # padding in the loss.
        if padding == "max_length" and data_args.ignore_pad_token_for_loss:
            labels["input_ids"] = [
                [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
            ]

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    if training_args.do_train:
        train_dataset = prepare_train_dataset(raw_datasets, data_args, training_args, preprocess_function, column_names)

    if training_args.do_eval:
        eval_dataset, max_target_length = prepare_eval_dataset(raw_datasets, data_args, training_args, preprocess_function, column_names)

    if training_args.do_predict:
        predict_dataset, max_target_length = prepare_predict_dataset(raw_datasets, data_args, training_args, preprocess_function, column_names)

    # Metric
    rouge = ROUGEMetric(tokenizer, data_args.ignore_pad_token_for_loss)

    # Initialize our Trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=rouge.compute_metrics if training_args.predict_with_generate else None,
    )

    # Training
    if training_args.do_train:
        logger.info(f"training data sample:\n{train_dataset[0]}")
        logger.info(f"detokenized input_ids:\n{tokenizer.batch_decode([train_dataset[0]['input_ids']], clean_up_tokenization_spaces=True)}")
        logger.info(f"detokenized labels:\n{tokenizer.batch_decode([train_dataset[0]['labels']], clean_up_tokenization_spaces=True)}")
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()  # Saves the tokenizer too for easy upload

        metrics = train_result.metrics
        max_train_samples = (
            data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # Evaluation
    results = {}
    max_length = (
        training_args.generation_max_length
        if training_args.generation_max_length is not None
        else data_args.val_max_target_length
    )
    num_beams = data_args.num_beams if data_args.num_beams is not None else training_args.generation_num_beams
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate(max_length=max_length, num_beams=num_beams, metric_key_prefix="eval")
        max_eval_samples = data_args.max_eval_samples if data_args.max_eval_samples is not None else len(eval_dataset)
        metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    if training_args.do_predict:
        logger.info("*** Predict ***")

        predict_results = trainer.predict(
            predict_dataset, metric_key_prefix="predict", max_length=max_length, num_beams=num_beams
        )
        metrics = predict_results.metrics
        max_predict_samples = (
            data_args.max_predict_samples if data_args.max_predict_samples is not None else len(predict_dataset)
        )
        metrics["predict_samples"] = min(max_predict_samples, len(predict_dataset))

        trainer.log_metrics("predict", metrics)
        trainer.save_metrics("predict", metrics)

        if trainer.is_world_process_zero():
            if training_args.predict_with_generate:
                predictions = tokenizer.batch_decode(
                    predict_results.predictions, skip_special_tokens=True, clean_up_tokenization_spaces=True
                )
                predictions = [pred.strip() for pred in predictions]
                output_prediction_file = os.path.join(training_args.output_dir, "generated_predictions.txt")
                with open(output_prediction_file, "w") as writer:
                    writer.write("\n".join(predictions))

    kwargs = {"finetuned_from": model_args.model_name_or_path, "tasks": "summarization"}
    if data_args.dataset_name is not None:
        kwargs["dataset_tags"] = data_args.dataset_name
        if data_args.dataset_config_name is not None:
            kwargs["dataset_args"] = data_args.dataset_config_name
            kwargs["dataset"] = f"{data_args.dataset_name} {data_args.dataset_config_name}"
        else:
            kwargs["dataset"] = data_args.dataset_name

    if data_args.lang is not None:
        kwargs["language"] = data_args.lang

    if training_args.push_to_hub:
        trainer.push_to_hub(**kwargs)
    else:
        trainer.create_model_card(**kwargs)

    return results


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()