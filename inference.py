import argparse
import os

from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor


def is_valid_file(parser, arg):
    if not os.path.exists(arg):
        parser.error(f"The file {arg} does not exist!")
    else:
        return arg


def main():
    parser = argparse.ArgumentParser(description='Process a user message and an image file path.')

    parser.add_argument(
        '-m', '--message',
        type=str,
        required=True,
        help='User message of type string'
    )

    parser.add_argument(
        '-i', '--image',
        type=lambda x: is_valid_file(parser, x),
        required=True,
        help='Path to the image file'
    )

    parser.add_argument(
        '--model_id_or_path',
        type=str,
        default="microsoft/Phi-3-vision-128k-instruct"
    )

    parser.add_argument(
        '--peft_model',
        type=str,
        default=None
    )

    args = parser.parse_args()

    print(f"User message: {args.message}")
    print(f"Image file path: {args.image}")

    model = AutoModelForCausalLM.from_pretrained(
        args.model_id_or_path,
        device_map="cuda",
        trust_remote_code=True,
        torch_dtype="auto",
        _attn_implementation='flash_attention_2'  # use _attn_implementation='eager' to disable flash attention
    )

    if args.peft_model is not None:
        model.load_adapter(args.peft_model)

    processor = AutoProcessor.from_pretrained(args.model_id_or_path, trust_remote_code=True)

    messages = [
        {"role": "user", "content": f"<|image_1|>\n{args.message}"}
    ]
    image = Image.open(args.image)

    prompt = processor.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    inputs = processor(prompt, [image], return_tensors="pt").to("cuda:0")

    generation_args = {
        "max_new_tokens": 10000,
        "temperature": 0.0,
        "do_sample": False,
    }

    generate_ids = model.generate(**inputs, eos_token_id=processor.tokenizer.eos_token_id, **generation_args)

    # remove input tokens
    generate_ids = generate_ids[:, inputs['input_ids'].shape[1]:]
    response = processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    print(response)


if __name__ == "__main__":
    main()
