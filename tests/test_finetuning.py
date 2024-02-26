# Copyright (c) Khulnasoft Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llmk 2 Community License Agreement.

import pytest
from pytest import approx
from unittest.mock import patch

import torch
from torch.optim import AdamW
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.sampler import BatchSampler

from llmk_recipes.finetuning import main
from llmk_recipes.data.sampler import LengthBasedBatchSampler


def get_fake_dataset():
    return [{
        "input_ids":[1],
        "attention_mask":[1],
        "labels":[1],
        }]


@patch('llmk_recipes.finetuning.train')
@patch('llmk_recipes.finetuning.LlmkForCausalLM.from_pretrained')
@patch('llmk_recipes.finetuning.LlmkTokenizer.from_pretrained')
@patch('llmk_recipes.finetuning.get_preprocessed_dataset')
@patch('llmk_recipes.finetuning.optim.AdamW')
@patch('llmk_recipes.finetuning.StepLR')
def test_finetuning_no_validation(step_lr, optimizer, get_dataset, tokenizer, get_model, train):
    kwargs = {"run_validation": False}

    get_dataset.return_value = get_fake_dataset()

    main(**kwargs)

    assert train.call_count == 1

    args, kwargs = train.call_args
    train_dataloader = args[1]
    eval_dataloader = args[2]

    assert isinstance(train_dataloader, DataLoader)
    assert eval_dataloader is None

    if torch.cuda.is_available():
        assert get_model.return_value.to.call_count == 1
        assert get_model.return_value.to.call_args.args[0] == "cuda"
    else:
        assert get_model.return_value.to.call_count == 0


@patch('llmk_recipes.finetuning.train')
@patch('llmk_recipes.finetuning.LlmkForCausalLM.from_pretrained')
@patch('llmk_recipes.finetuning.LlmkTokenizer.from_pretrained')
@patch('llmk_recipes.finetuning.get_preprocessed_dataset')
@patch('llmk_recipes.finetuning.optim.AdamW')
@patch('llmk_recipes.finetuning.StepLR')
def test_finetuning_with_validation(step_lr, optimizer, get_dataset, tokenizer, get_model, train):
    kwargs = {"run_validation": True}

    get_dataset.return_value = get_fake_dataset()

    main(**kwargs)

    assert train.call_count == 1

    args, kwargs = train.call_args
    train_dataloader = args[1]
    eval_dataloader = args[2]
    assert isinstance(train_dataloader, DataLoader)
    assert isinstance(eval_dataloader, DataLoader)

    if torch.cuda.is_available():
        assert get_model.return_value.to.call_count == 1
        assert get_model.return_value.to.call_args.args[0] == "cuda"
    else:
        assert get_model.return_value.to.call_count == 0


@patch('llmk_recipes.finetuning.train')
@patch('llmk_recipes.finetuning.LlmkForCausalLM.from_pretrained')
@patch('llmk_recipes.finetuning.LlmkTokenizer.from_pretrained')
@patch('llmk_recipes.finetuning.get_preprocessed_dataset')
@patch('llmk_recipes.finetuning.generate_peft_config')
@patch('llmk_recipes.finetuning.get_peft_model')
@patch('llmk_recipes.finetuning.optim.AdamW')
@patch('llmk_recipes.finetuning.StepLR')
def test_finetuning_peft(step_lr, optimizer, get_peft_model, gen_peft_config, get_dataset, tokenizer, get_model, train):
    kwargs = {"use_peft": True}

    get_dataset.return_value = get_fake_dataset()

    main(**kwargs)

    if torch.cuda.is_available():
        assert get_model.return_value.to.call_count == 1
        assert get_model.return_value.to.call_args.args[0] == "cuda"
    else:
        assert get_model.return_value.to.call_count == 0
    
    assert get_peft_model.return_value.print_trainable_parameters.call_count == 1


@patch('llmk_recipes.finetuning.train')
@patch('llmk_recipes.finetuning.LlmkForCausalLM.from_pretrained')
@patch('llmk_recipes.finetuning.LlmkTokenizer.from_pretrained')
@patch('llmk_recipes.finetuning.get_preprocessed_dataset')
@patch('llmk_recipes.finetuning.get_peft_model')
@patch('llmk_recipes.finetuning.StepLR')
def test_finetuning_weight_decay(step_lr, get_peft_model, get_dataset, tokenizer, get_model, train, mocker):
    kwargs = {"weight_decay": 0.01}

    get_dataset.return_value = get_fake_dataset()
    
    model = mocker.MagicMock(name="Model")
    model.parameters.return_value = [torch.ones(1,1)]

    get_model.return_value = model 

    main(**kwargs)

    assert train.call_count == 1

    args, kwargs = train.call_args
    optimizer = args[4]

    print(optimizer.state_dict())

    assert isinstance(optimizer, AdamW)
    assert optimizer.state_dict()["param_groups"][0]["weight_decay"] == approx(0.01)


@patch('llmk_recipes.finetuning.train')
@patch('llmk_recipes.finetuning.LlmkForCausalLM.from_pretrained')
@patch('llmk_recipes.finetuning.LlmkTokenizer.from_pretrained')
@patch('llmk_recipes.finetuning.get_preprocessed_dataset')
@patch('llmk_recipes.finetuning.optim.AdamW')
@patch('llmk_recipes.finetuning.StepLR')
def test_batching_strategy(step_lr, optimizer, get_dataset, tokenizer, get_model, train):
    kwargs = {"batching_strategy": "packing"}

    get_dataset.return_value = get_fake_dataset()

    main(**kwargs)

    assert train.call_count == 1

    args, kwargs = train.call_args
    train_dataloader, eval_dataloader = args[1:3]
    assert isinstance(train_dataloader.batch_sampler, BatchSampler)
    assert isinstance(eval_dataloader.batch_sampler, BatchSampler)

    kwargs["batching_strategy"] = "padding"
    train.reset_mock()
    main(**kwargs)

    assert train.call_count == 1

    args, kwargs = train.call_args
    train_dataloader, eval_dataloader = args[1:3]
    assert isinstance(train_dataloader.batch_sampler, LengthBasedBatchSampler)
    assert isinstance(eval_dataloader.batch_sampler, LengthBasedBatchSampler)

    kwargs["batching_strategy"] = "none"

    with pytest.raises(ValueError):
        main(**kwargs)
