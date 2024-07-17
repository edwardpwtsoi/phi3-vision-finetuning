import argparse
import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Tuple

import Levenshtein
import torch
from accelerate import Accelerator
from datasets import load_dataset
from peft import LoraConfig
from transformers import (
    AutoModelForCausalLM,
    AutoProcessor,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
)

from evaluation import evaluate


# suggested deepspeed config
DS_CONFIG_DICT = {
    'zero_optimization': {
        'stage': 2,
        'allgather_partitions': True,
        'allgather_bucket_size': 5e8,
        'overlap_comm': True,
        'reduce_scatter': True,
        'reduce_bucket_size': 5e8,
        'contiguous_gradients': True,
        'round_robin_gradients': True,
    },
    'fp16': {
        'enabled': 'auto',
        'loss_scale': 0,
        'loss_scale_window': 1000,
        'initial_scale_power': 16,
        'hysteresis': 2,
        'min_loss_scale': 1,
    },
    'bf16': {'enabled': 'auto'},
    'train_micro_batch_size_per_gpu': 'auto',
    'train_batch_size': 'auto',
    'gradient_accumulation_steps': 'auto',
    'gradient_clipping': 'auto',
}


def create_dataset(use_full_train=False):
    """
    PubTabNet-HTML dataset from the Hugging Face Hub
    """
    train_dataset = load_dataset('apoidea/pubtabnet-html', split='train')
    eval_dataset = load_dataset('apoidea/pubtabnet-html', split='validation')

    return train_dataset, eval_dataset


def create_lora_config(rank, alpha_to_rank_ratio=2.0, dropout=0.0):
    linear_modules = [
        # CLIP modules
        # 'q_proj',  # attention
        # 'k_proj',
        # 'v_proj',
        # 'out_proj',
        # 'fc1',  # MLP
        # 'fc2',
        # 'img_projection.0',
        # 'img_projection.2',
        # FIXME: can't lora CLIP is a known issue of Phi-3-V
        # Phi language modules
        'qkv_proj',  # attention
        'o_proj',
        'down_proj',  # MLP
        'gate_up_proj',
        'lm_head',
    ]
    lora_config = LoraConfig(
        r=rank,
        lora_alpha=round(rank * alpha_to_rank_ratio),
        lora_dropout=dropout,
        target_modules=linear_modules,
        init_lora_weights='gaussian',
    )
    return lora_config


class NoGradHook:
    def __init__(self):
        self.prev_enabled = True

    def maybe_enable_grad_hook(self, *_):
        torch.set_grad_enabled(self.prev_enabled)

    def disable_grad_hook(self, *_):
        self.prev_enabled = torch.is_grad_enabled()
        torch.set_grad_enabled(False)


def freeze_vision_model(model):
    vision_no_grad_hook = NoGradHook()
    vision_module = model.model.vision_embed_tokens
    vision_module.register_forward_pre_hook(vision_no_grad_hook.disable_grad_hook)
    vision_module.register_forward_hook(vision_no_grad_hook.maybe_enable_grad_hook)
    for p in vision_module.parameters():
        p.requires_grad_(False)


def create_model(model_name_or_path, use_flash_attention=False, use_qlora=False):
    bnb_config = (
        BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type='nf4',
            bnb_4bit_compute_dtype=torch.bfloat16 if use_flash_attention else torch.float16,
        )
        if use_qlora
        else None
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        # Phi-3-V is originally trained in bf16 + flash attn
        # For fp16 mixed precision training, load in f32 to avoid hf accelerate error
        torch_dtype=torch.bfloat16 if use_flash_attention else torch.float32,
        trust_remote_code=True,
        _attn_implementation='flash_attention_2' if use_flash_attention else 'eager',
        quantization_config=bnb_config,
    )

    return model


class VQADataCollatorBase(ABC):
    def __init__(self, processor):
        self.processor = processor

    @staticmethod
    @abstractmethod
    def _get_image_from_example(example):
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def _get_question_and_answer_from_example(example) -> Tuple[str, str]:
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def _form_prompt_message(question):
        raise NotImplementedError

    def __call__(self, examples):
        assert len(examples) == 1, f'Phi-3-V only supports batch_size == 1, but got examples of length {len(examples)}'
        example = examples[0]

        image = self._get_image_from_example(example)
        question, answer = self._get_question_and_answer_from_example(example)
        prompt_message = self._form_prompt_message(question)
        prompt = self.processor.tokenizer.apply_chat_template(
            [prompt_message], tokenize=False, add_generation_prompt=True
        )
        answer = f'{answer}<|end|>\n<|endoftext|>'

        # mask questions for labels
        batch = self.processor(prompt, [image], return_tensors='pt')
        prompt_input_ids = batch['input_ids']
        # Do not add bos token to answer
        answer_input_ids = self.processor.tokenizer(
            answer, add_special_tokens=False, return_tensors='pt'
        )['input_ids']
        input_ids = torch.cat([prompt_input_ids, answer_input_ids], dim=1)
        ignore_index = -100
        labels = torch.cat(
            [
                torch.tensor([ignore_index] * len(prompt_input_ids[0])).unsqueeze(0),
                answer_input_ids,
            ],
            dim=1,
        )

        batch['input_ids'] = input_ids
        del batch['attention_mask']
        batch['labels'] = labels

        return batch


class PubTabNetHTMLDataCollator(VQADataCollatorBase):
    @staticmethod
    def _get_image_from_example(example):
        return example['image']

    @staticmethod
    def _get_question_and_answer_from_example(example):
        return "Reconstruct the table in the image in a HTML format.", example["html_table"]

    @staticmethod
    def _form_prompt_message(question):
        return {
            'role': 'user',
            'content': f'<|image_1|>\n{question}',
        }


def normalized_levenshtein(s1, s2):
    len_s1, len_s2 = len(s1), len(s2)
    distance = Levenshtein.distance(s1, s2)
    return distance / max(len_s1, len_s2)


def similarity_score(a_ij, o_q_i, tau=0.5):
    nl = normalized_levenshtein(a_ij, o_q_i)
    return 1 - nl if nl < tau else 0


def average_normalized_levenshtein_similarity(ground_truth, predicted_answers):
    assert len(ground_truth) == len(
        predicted_answers
    ), 'Length of ground_truth and predicted_answers must match.'

    N = len(ground_truth)
    total_score = 0

    for i in range(N):
        a_i = ground_truth[i]
        o_q_i = predicted_answers[i]
        if o_q_i == '':
            print('Warning: Skipped an empty prediction.')
            max_score = 0
        else:
            max_score = max(similarity_score(a_ij, o_q_i) for a_ij in a_i)

        total_score += max_score

    return total_score / N


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model_name_or_path',
        type=str,
        default='microsoft/Phi-3-vision-128k-instruct',
        help='Model name or path to load from',
    )
    parser.add_argument(
        '--full_train', action='store_true', help='Use full training dataset (DocVQA)'
    )
    parser.add_argument('--pre_evaluation', action='store_true', help='Evaluate before finetune')
    parser.add_argument('--use_flash_attention', action='store_true', help='Use Flash Attention')
    parser.add_argument('--bf16', action='store_true', help='Use BF16')
    parser.add_argument('--use_lora', action='store_true', help='Use LoRA')
    parser.add_argument('--use_qlora', action='store_true', help='Use QLora')
    parser.add_argument('--output_dir', type=str, default='./output/', help='Output directory')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--num_crops', type=int, default=16, help='Number of maximum image crops')
    parser.add_argument('--max_steps', type=int, default=-1, help='Number of maximum training steps')
    parser.add_argument('--warmup_steps', type=int, default=10, help='Number of warmup steps')
    parser.add_argument(
        '--num_train_epochs', type=int, default=1, help='Number of training epochs'
    )
    parser.add_argument('--learning_rate', type=float, default=4.0e-5, help='Learning rate')
    parser.add_argument('--wd', type=float, default=0.01, help='Weight decay')
    parser.add_argument('--no-tqdm', dest='tqdm', action='store_false', help='Disable tqdm')
    parser.add_argument('--lora_rank', type=int, default=64, help='LoRA rank')
    parser.add_argument(
        '--lora_alpha_ratio', type=float, default=2, help='LoRA alpha to rank ratio'
    )
    parser.add_argument('--lora_dropout', type=float, default=0.0, help='LoRA dropout')
    parser.add_argument('--freeze_vision_model', action='store_true', help='Freeze vision model')
    args = parser.parse_args()

    assert args.num_crops <= 16, 'num_crops must be less than or equal to 16'
    if args.use_qlora:
        args.use_lora = True

    accelerator = Accelerator()

    with accelerator.local_main_process_first():
        processor = AutoProcessor.from_pretrained(
            args.model_name_or_path, trust_remote_code=True, num_crops=args.num_crops
        )
        model = create_model(
            args.model_name_or_path,
            use_flash_attention=args.use_flash_attention,
            use_qlora=args.use_qlora,
        )

    train_dataset, eval_dataset = create_dataset(use_full_train=args.full_train)

    num_gpus = accelerator.num_processes
    print(f'training on {num_gpus} GPUs')
    assert args.batch_size % num_gpus == 0, 'Batch size must be divisible by the number of GPUs'
    gradient_accumulation_steps = args.batch_size // num_gpus
    if args.bf16:
        fp16 = False
        bf16 = True
    else:
        fp16 = True
        bf16 = False

    # hard coded training args
    training_args = TrainingArguments(
        num_train_epochs=args.num_train_epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=1,  # NOTE currently only supports batch_size == 1
        per_device_eval_batch_size=1,
        gradient_checkpointing=True,
        gradient_accumulation_steps=gradient_accumulation_steps,
        optim='adamw_torch',
        adam_beta1=0.9,
        adam_beta2=0.95,
        adam_epsilon=1e-7,
        learning_rate=args.learning_rate,
        weight_decay=args.wd,
        max_grad_norm=1.0,
        lr_scheduler_type='linear',
        warmup_steps=args.warmup_steps,
        logging_steps=10,
        output_dir=args.output_dir,
        save_strategy='steps',
        save_steps=0.1,
        save_total_limit=10,
        save_only_model=True,
        bf16=bf16,
        fp16=fp16,
        remove_unused_columns=False,
        report_to='none',
        deepspeed=None if args.use_lora else DS_CONFIG_DICT,
        disable_tqdm=not args.tqdm,
        dataloader_num_workers=4,
        dataloader_prefetch_factor=2,
    )

    data_collator = PubTabNetHTMLDataCollator(processor)

    # eval before fine-tuning
    out_path = Path(training_args.output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    if not args.use_qlora:
        local_rank = int(os.environ.get('LOCAL_RANK', 0))
        model = model.to(f'cuda:{local_rank}')
    if args.pre_evaluation:
        anls = evaluate(
            model=model,
            processor=processor,
            dataset=eval_dataset,
            output=out_path / 'eval_before',
            disable_tqdm=not args.tqdm,
        )
        if accelerator.is_main_process:
            print(f'Average normalized Levenshtein similarity before finetuning: {anls}')

    if args.use_lora:
        lora_config = create_lora_config(
            rank=args.lora_rank,
            alpha_to_rank_ratio=args.lora_alpha_ratio,
            dropout=args.lora_dropout,
        )
        model.add_adapter(lora_config)
        model.enable_adapters()

        # NOTE: cannot train vision model with LoRA is a known issue of Phi-3-V
        args.freeze_vision_model = True

    if args.freeze_vision_model:
        freeze_vision_model(model)

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
    )
    trainer.train()
    trainer.save_model()
    if accelerator.is_main_process:
        processor.save_pretrained(training_args.output_dir)
    accelerator.wait_for_everyone()

    # eval after fine-tuning
    if args.use_lora:
        # first try to clear GPU memory
        del model
        del trainer
        __import__('gc').collect()
        torch.cuda.empty_cache()

        # reload the model for inference
        # this part also serves as an example of how to load a trained model
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path,
            # Phi-3-V is originally trained in bf16 + flash attn
            # For fp16 mixed precision training, load in f32 to avoid hf accelerate error
            torch_dtype=torch.bfloat16 if args.use_flash_attention else torch.float32,
            trust_remote_code=True,
            _attn_implementation='flash_attention_2' if args.use_flash_attention else 'eager',
        )
        model.load_adapter(training_args.output_dir)
    else:
        # for full finetuning, GPU memory can't be cleared (likely caused by deepspeed
        # https://github.com/microsoft/DeepSpeed/issues/3677)
        # so we don't reload the model
        model = accelerator.unwrap_model(model, keep_fp32_wrapper=not args.bf16)

        # below is a sample code snippet to load fully-finetuned model
        # model = AutoModelForCausalLM.from_pretrained(
        #     training_args.output_dir,
        #     # Phi-3-V is originally trained in bf16 + flash attn
        #     # For fp16 mixed precision training, load in f32 to avoid hf accelerate error
        #     torch_dtype=torch.bfloat16 if args.use_flash_attention else torch.float32,
        #     trust_remote_code=True,
        #     _attn_implementation='flash_attention_2' if args.use_flash_attention else 'eager',
        # )

    rank = int(os.environ.get('RANK', 0))
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    model = model.to(f'cuda:{local_rank}')
    anls = evaluate(
        model=model,
        processor=processor,
        dataset=eval_dataset,
        save_path=out_path / 'eval_after.json',
        disable_tqdm=not args.tqdm,
    )
    if rank == 0:
        print(f'Average normalized Levenshtein similarity after finetuning: {anls}')


if __name__ == '__main__':
    main()
