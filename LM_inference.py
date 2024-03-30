from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
import json
from tqdm import tqdm
from argparse import ArgumentParser
import os


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--setting", type=str, required=True)
    parser.add_argument("--model", type=str, default='deepseek-7b')
    parser.add_argument("--output_dir", type=str)
    parser.add_argument("--moda", type=str, default='greedy')
    parser.add_argument("--max_tokens", type=int, default=500)
    return parser.parse_args()

def load_model(model_name: str):
    if model_name.startswith("deepseek-7b"):
        print("Loading deepseek-coder-7b")
        model_dir = "deepseek-ai/deepseek-coder-6.7b-base"
    elif model_name.startswith("deepseek-33b"):
        print("Loading deepseek-coder-33b")
        model_dir = "deepseek-ai/deepseek-coder-33b-base"
    elif model_name.startswith("codellama-7b"):
        print("Loading codellama-7b")
        model_dir = "codellama/CodeLlama-7b-Python-hf"
    elif model_name.startswith("codellama-13b"):
        print("Loading codellama-13b")
        model_dir = "codellama/CodeLlama-13b-Python-hf"
    elif model_name.startswith("starcoder2-7b"):
        print("Loading starcoder2-7b")
        model_dir = "bigcode/starcoder2-7b"
    elif model_name.startswith("starcoder2-15b"):
        print("Loading starcoder2-15b")
        model_dir = "bigcode/starcoder2-15b"
    elif model_name == "gemma-7b":
        print("Loading gemma-7b")
        model_dir = "google/gemma-7b"
    elif model_name == "qwen1.5-7b":
        print("Loading qwen1.5-7b")
        model_dir = "Qwen/Qwen1.5-7B"
    model = LLM(model=model_dir, trust_remote_code=True, gpu_memory_utilization=0.9, tensor_parallel_size=4)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    return model, tokenizer


def retrieve_context_length(model_name: str):
    if model_name.startswith("deepseek") or model_name.startswith("codellama") or model_name.startswith("starcoder2"):
        return 16384
    elif model_name.startswith("gemma-7b"):
        return 8192
    elif model_name.startswith("qwen1.5-7b"):
        return 32768


def retrieve_special_ids(model_name: str, tokenizer):
    if model_name.startswith("qwen1.5-7b"):
        bos_id = 151643
    else:
        bos_id = tokenizer.bos_token_id

    if model_name.startswith("deepseek"):
        prefix_id = tokenizer.convert_tokens_to_ids("<｜fim▁begin｜>")
        middle_id = tokenizer.convert_tokens_to_ids("<｜fim▁hole｜>")
        suffix_id = tokenizer.convert_tokens_to_ids("<｜fim▁end｜>")
    elif model_name.startswith("codellama"):
        prefix_id = tokenizer.prefix_id
        middle_id = tokenizer.middle_id
        suffix_id = tokenizer.suffix_id
    elif model_name.startswith("starcoder2"):
        prefix_id = tokenizer.convert_tokens_to_ids("<fim_prefix>")
        middle_id = tokenizer.convert_tokens_to_ids("<fim_middle>")
        suffix_id = tokenizer.convert_tokens_to_ids("<fim_suffix>")
    else:
        prefix_id, middle_id, suffix_id = None, None, None

    return bos_id, prefix_id, middle_id, suffix_id


def produce_prompt(args, task, js, tokenizer):
    context_window = retrieve_context_length(args.model)
    input_code = js['input_code']
    assert input_code is not None
    input_code_ids = tokenizer(input_code)['input_ids']
    bos_id, prefix_id, middle_id, suffix_id = retrieve_special_ids(args.model, tokenizer)
    if len(input_code_ids) > context_window:
        raise ValueError(f"Input code is too long: {len(input_code_ids)} tokens")
    max_context_length = context_window - len(input_code_ids) - args.max_tokens
    if task == 'baseline':
        if js['class_name']:
            input_code = f"class js['class_name']:\n{input_code}"
            input_code_ids = tokenizer(input_code)['input_ids']
        prompt_ids = [bos_id] + input_code_ids
    elif task == 'local_completion': # local file (completion)
        context_above = js['contexts_above']
        assert context_above is not None
        context_above_ids = tokenizer(context_above)['input_ids']
        if len(context_above_ids) > max_context_length:
            context_above_ids = context_above_ids[-max_context_length:]
        prompt_ids = [bos_id] + context_above_ids + input_code_ids
    elif task == 'local_infilling': # local file (hole)
        context_above, context_below = js['contexts_above'], js['contexts_below']
        assert context_above is not None and context_below is not None
        assert prefix_id is not None and middle_id is not None and suffix_id is not None
        context_above_ids = tokenizer(context_above)['input_ids']
        context_below_ids = tokenizer(context_below)['input_ids']
        if len(context_above_ids) + len(context_below_ids) > max_context_length:
            context_above_ratio = len(context_above_ids) / (len(context_above_ids) + len(context_below_ids))
            context_below_ratio = len(context_below_ids) / (len(context_above_ids) + len(context_below_ids))
            max_context_above_length = int(max_context_length * context_above_ratio)
            max_context_below_length = int(max_context_length * context_below_ratio)
            context_above_ids = context_above_ids[-max_context_above_length:]
            context_below_ids = context_below_ids[:max_context_below_length]
        if args.model.startswith('deepseek'):
            prompt_ids = [bos_id, prefix_id] + context_above_ids + input_code_ids + [middle_id] + context_below_ids + [suffix_id]
        else:
            prompt_ids = [bos_id, prefix_id] + context_above_ids + input_code_ids + [suffix_id] + context_below_ids + [middle_id]
    else:
        raise ValueError("Invalid context")     

    return prompt_ids


def load_finished_data(output_file):
    finished_data = []
    if not os.path.exists(output_file):
        return finished_data
    with open(output_file, 'r') as f:
        for line in f:
            js = json.loads(line)
            finished_data.append(js['namespace'])
    return finished_data


def inference(args, task, model, tokenizer, prompt_file, output_dir, sampling_params):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_file = os.path.join(output_dir, 'completion.jsonl')
    finished_data = load_finished_data(output_file)
    with open(prompt_file, 'r') as f:
        f = f.readlines()
        with open(output_file, 'a') as f_out:
            for line in tqdm(f):
                js = json.loads(line)
                if js['namespace'] in finished_data:
                    continue
                prompt_ids = produce_prompt(args, task, js, tokenizer)
                
                try:
                    results = model.generate(prompt_token_ids=[prompt_ids], sampling_params=sampling_params, use_tqdm=False)
                    completions = []
                    for result in results:
                        for output in result.outputs:
                            completions.append(output.text)
                except Exception as e:
                    print(f"Error: {e}")
                    continue
                cases = {'namespace': js['namespace'], 'completions': completions}
                f_out.write(json.dumps(cases) + '\n')
                f_out.flush()


def main():
    args = parse_args()
    model, tokenizer = load_model(args.model)
    print("Loaded model and tokenizer.")

    if args.moda == 'greedy':
        sampling_param = SamplingParams(temperature=0.0, max_tokens=args.max_tokens, n=1)
    elif args.moda == 'sampling':
        sampling_param = SamplingParams(temperature=0.4, top_p=0.95, max_tokens=args.max_tokens, n=20)
    else:
        raise ValueError("Invalid moda")

    inference(args, args.setting, model, tokenizer, 
              'prompt/prompt_elements.jsonl', 
              args.output_dir, 
              sampling_param
    )
    
if __name__ == '__main__':
    main()