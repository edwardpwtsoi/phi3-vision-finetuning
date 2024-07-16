import argparse
import os

import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor


def is_valid_file(parser, arg):
    if not os.path.exists(arg):
        parser.error(f"The file {arg} does not exist!")
    else:
        return arg


class Phi3VisionPredictor:
    def __init__(self, model_id_or_path, use_flash_attn=True, peft_model=None, use_torch_compile=True):
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id_or_path,
            device_map="cuda",
            trust_remote_code=True,
            torch_dtype="auto",
            _attn_implementation='flash_attention_2' if use_flash_attn else 'eager'
        )

        if peft_model is not None:
            self.model.load_adapter(peft_model)

        self.model.eval()

        if use_torch_compile:
            self.model = torch.compile(self.model)

        self.processor = AutoProcessor.from_pretrained(model_id_or_path, trust_remote_code=True)

    @torch.no_grad()
    def __call__(self, message, image=None):
        messages = [
            {"role": "user", "content": f"<|image_1|>\n{message}"}
        ]

        prompt = self.processor.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        inputs = self.processor(prompt, [image], return_tensors="pt").to("cuda:0")

        generation_args = {
            "max_new_tokens": 10000,
            "temperature": 0.0,
            "do_sample": False,
        }

        generate_ids = self.model.generate(**inputs, eos_token_id=self.processor.tokenizer.eos_token_id, **generation_args)

        # remove input tokens
        generate_ids = generate_ids[:, inputs['input_ids'].shape[1]:]
        response = self.processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

        return response


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
    image = Image.open(args.image)

    predictor = Phi3VisionPredictor(args.model_id_or_path, peft_model=args.peft_model)
    response = predictor(args.message, image)
    print(response)


if __name__ == "__main__":
    main()
