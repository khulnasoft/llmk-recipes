# Copyright (c) Khulnasoft Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llmk 2 Community License Agreement.

import functools

from transformers.models.llmk.modeling_llmk import LlmkDecoderLayer
from torch.distributed.fsdp.wrap import (
    transformer_auto_wrap_policy,
    size_based_auto_wrap_policy,
)


def get_size_policy(min_params=1e8):
    num_wrap_policy = functools.partial(
        size_based_auto_wrap_policy, min_num_params=min_params
    )
    return num_wrap_policy


def get_llmk_wrapper():
    """we register our main layer class and use the fsdp transformer wrapping policy
    ensures embedding layers are in the root fsdp unit for shared access and that fsdp units map to transformer layers
    """
    # ====   use new transformer wrapper

    llmk_auto_wrap_policy = functools.partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls={
            LlmkDecoderLayer,
        },
    )

    return llmk_auto_wrap_policy
