# Examples

This folder contains finetuning and inference examples for Llmk 2, Code Llmk and (Purple Llmk](https://ai.khulnasoft.com/llmk/purple-llmk/). For the full documentation on these examples please refer to [docs/inference.md](../docs/inference.md)

## Finetuning

Please refer to the main [README.md](../README.md) for information on how to use the [finetuning.py](./finetuning.py) script.
After installing the llmk-recipes package through [pip](../README.md#installation) you can also invoke the finetuning in two ways:
```
python -m llmk_recipes.finetuning <parameters>

python examples/finetuning.py <parameters>
```
Please see [README.md](../README.md) for details.

## Inference 
So far, we have provide the following inference examples:

1. [inference script](./inference.py) script provides support for Hugging Face accelerate, PEFT and FSDP fine tuned models. It also demonstrates safety features to protect the user from toxic or harmful content.

2. [vllm/inference.py](./vllm/inference.py) script takes advantage of vLLM's paged attention concept for low latency.

3. The [hf_text_generation_inference](./hf_text_generation_inference/README.md) folder contains information on Hugging Face Text Generation Inference (TGI).

4. A [chat completion](./chat_completion/chat_completion.py) example highlighting the handling of chat dialogs.

5. [Code Llmk](./code_llmk/) folder which provides examples for [code completion](./code_llmk/code_completion_example.py), [code infilling](./code_llmk/code_infilling_example.py) and [Llmk2 70B code instruct](./code_llmk/code_instruct_example.py).

6. The [Purple Llmk Using Anyscale](./Purple_Llmk_Anyscale.ipynb) is a notebook that shows how to use Anyscale hosted Llmk Guard model to classify user inputs as safe or unsafe.

7. [Llmk Guard](./llmk_guard/) inference example and [safety_checker](../src/llmk_recipes/inference/safety_utils.py) for the main [inference](./inference.py) script. The standalone scripts allows to test Llmk Guard on user input, or user input and agent response pairs. The safety_checker integration providers a way to integrate Llmk Guard on all inference executions, both for the user input and model output.

For more in depth information on inference including inference safety checks and examples, see the inference documentation [here](../docs/inference.md).

**Note** The [sensitive topics safety checker](../src/llmk_recipes/inference/safety_utils.py) utilizes AuditNLG which is an optional dependency. Please refer to installation section of the main [README.md](../README.md#install-with-optional-dependencies) for details.

**Note** The **vLLM** example requires additional dependencies. Please refer to installation section of the main [README.md](../README.md#install-with-optional-dependencies) for details.

## Train on custom dataset
To show how to train a model on a custom dataset we provide an example to generate a custom dataset in [custom_dataset.py](./custom_dataset.py).
The usage of the custom dataset is further described in the datasets [README](../docs/Dataset.md#training-on-custom-data).
