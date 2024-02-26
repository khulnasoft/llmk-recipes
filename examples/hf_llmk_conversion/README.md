# Convert Hugging Face llmk weights to official llmk consolidated format

This is the reverse conversion for `convert_llmk_weights_to_hf.py` script from the transformer package.

## Step 0: Convert to consolidated format
- Create an output directory for the converted weights, such as `test70B`.
- Copy file params.json from the official llmk download into that directory.
- Run the conversion script. `model-path` can be a Hugging Face hub model or a local hf model directory.
```
python -m llmk_recipes.tools.convert_hf_weights_to_llmk --model-path khulnasoft-llmk/Llmk-2-70b-chat-hf --output-dir test70B --model-size 70B
```

## Step 1: Run inference
Checkout the official llmk inference [repo](https://github.com/khulnasoft/llmk). Test using chat or text completion.
```
torchrun --nproc_per_node 8 example_chat_completion.py --ckpt_dir ./test70B --tokenizer_path ${llmk_2_dir}/tokenizer.model
```

For validation, please compare the converted weights with official llmk 2 weights
```
python compare_llmk_weights.py test70B ${llmk_2_70b_chat_dir}
```
