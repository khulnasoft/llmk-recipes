# Copyright (c) Khulnasoft Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llmk 2 Community License Agreement.

import pytest
from functools import partial
from unittest.mock import patch


@pytest.mark.skip_missing_tokenizer
@patch('llmk_recipes.finetuning.train')
@patch('llmk_recipes.finetuning.LlmkTokenizer')
@patch('llmk_recipes.finetuning.LlmkForCausalLM.from_pretrained')
@patch('llmk_recipes.finetuning.optim.AdamW')
@patch('llmk_recipes.finetuning.StepLR')
def test_samsum_dataset(step_lr, optimizer, get_model, tokenizer, train, mocker, setup_tokenizer):
    from llmk_recipes.finetuning import main

    setup_tokenizer(tokenizer)

    BATCH_SIZE = 8
    kwargs = {
        "model_name": "khulnasoft-llmk/Llmk-2-7b-hf",
        "batch_size_training": BATCH_SIZE,
        "val_batch_size": 1,
        "use_peft": False,
        "dataset": "samsum_dataset",
        "batching_strategy": "padding",
        }

    main(**kwargs)

    assert train.call_count == 1

    args, kwargs = train.call_args
    train_dataloader = args[1]
    eval_dataloader = args[2]

    VAL_SAMPLES = 818
    TRAIN_SAMPLES = 14732

    assert len(train_dataloader) == TRAIN_SAMPLES // BATCH_SIZE
    assert len(eval_dataloader) == VAL_SAMPLES

    batch = next(iter(train_dataloader))

    assert "labels" in batch.keys()
    assert "input_ids" in batch.keys()
    assert "attention_mask" in batch.keys()

    assert batch["labels"][0][268] == -100
    assert batch["labels"][0][269] == 319

    assert batch["input_ids"][0][0] == 1
    assert batch["labels"][0][-1] == 2
    assert batch["input_ids"][0][-1] == 2
