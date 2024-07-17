import argparse
import os
import time

import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor


def is_valid_file(parser, arg):
    if not os.path.exists(arg):
        parser.error(f"The file {arg} does not exist!")
    else:
        return arg


class Phi3VisionPredictor:
    def __init__(self, model_id_or_path=None, model=None, processor=None, use_flash_attn=True, peft_model=None, use_torch_compile=True, local_rank=0):
        self.local_rank = local_rank
        if model_id_or_path is not None:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_id_or_path,
                device_map=f"cuda:{local_rank}",
                trust_remote_code=True,
                torch_dtype=torch.bfloat16 if use_flash_attn else torch.float32,
                _attn_implementation='flash_attention_2' if use_flash_attn else 'eager'
            )
            self.processor = AutoProcessor.from_pretrained(model_id_or_path, trust_remote_code=True)
        elif model is not None and processor is not None:
            self.model = model.to(f"cuda:{local_rank}")
            self.processor = processor
        else:
            raise ValueError("Either `model_id_or_path` or `model and processor` should be provided")

        if peft_model is not None:
            self.model.load_adapter(peft_model)

        self.model.eval()

        if use_torch_compile:
            self.model = torch.compile(self.model)

    @torch.no_grad()
    def __call__(self, message, image=None, max_new_tokens=5000, do_sample=False):
        messages = [
            {"role": "user", "content": f"<|image_1|>\n{message}"}
        ]

        prompt = self.processor.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        inputs = self.processor(prompt, [image], return_tensors="pt").to(f"cuda:{self.local_rank}")

        generate_ids = self.model.generate(
            **inputs,
            eos_token_id=self.processor.tokenizer.eos_token_id,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample
        )

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

    parser.add_argument(
        '--max_new_tokens',
        type=int,
        default=5000
    )

    parser.add_argument(
        '--do_sample',
        action="store_false"
    )

    args = parser.parse_args()

    print(f"User message: {args.message}")
    print(f"Image file path: {args.image}")
    image = Image.open(args.image)

    predictor = Phi3VisionPredictor(args.model_id_or_path, peft_model=args.peft_model)
    then = time.time()
    response = predictor(args.message, image, args.max_new_tokens, args.do_sample)
    print(response)
    print(f"Time Taken: {time.time() - then}")


if __name__ == "__main__":
    main()
