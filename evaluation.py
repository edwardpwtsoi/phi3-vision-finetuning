import argparse
import json
import os
import random

import torch
from accelerate import Accelerator
from accelerate.utils import gather_object
from datasets import load_dataset
from table_recognition_metric import TEDS
from tqdm import tqdm

from inference import Phi3VisionPredictor


@torch.no_grad()
def evaluate(model_id_or_path=None, dataset_id=None, *, model=None, processor=None, dataset=None,
             peft_model=None, split="validation", subsampling=1.0, output_path=None, disable_tqdm=False):
    if model_id_or_path is None and (model is None or processor is None):
        raise ValueError("Either `model_id_or_path` or `model` & `processor` must be provided")
    if dataset_id is None and dataset is None:
        raise ValueError("Either `dataset_id` or `dataset` must be provided")
    rank = int(os.environ.get('RANK', 0))
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    world_size = int(os.environ.get('WORLD_SIZE', 1))

    accelerator = Accelerator()
    teds = TEDS()

    with accelerator.local_main_process_first():
        predictor = Phi3VisionPredictor(model_id_or_path, model, processor, peft_model=peft_model, local_rank=local_rank)
        dataset = load_dataset(dataset_id, split=split) if dataset is None else dataset
        if 0. < subsampling < 1.:
            sampling_size = int(len(dataset) * subsampling)
            indices = list(range(len(dataset)))
            random.shuffle(indices)
            selector = indices[:sampling_size]
            dataset = dataset.select(selector)

    answers_unique = []
    generated_texts_unique = []

    dataset_shard = dataset.shard(num_shards=world_size, index=rank)
    for i in tqdm(range(len(dataset_shard)), disable=(rank != 0) or disable_tqdm):
        # Phi-3-V currently only supports batch_size == 1
        example = dataset_shard[i]
        answers_unique.append(example['html_table'])
        image = example['image']
        question = "Reconstruct the table in the image in a HTML format."
        response = predictor(question, image)
        generated_texts_unique.extend(response)

    generated_texts_unique = [g.strip().strip('.') for g in generated_texts_unique]

    # gather outputs from all ranks
    answers_unique = gather_object(answers_unique)
    generated_texts_unique = gather_object(generated_texts_unique)

    if accelerator.is_main_process:
        scores = [teds(gt_html, pred_html) for gt_html, pred_html in tqdm(zip(answers_unique, generated_texts_unique))]
        average_scores = sum(scores) / len(scores)
        print(f"Evaluation: {average_scores}")
        if output_path is None:
            output_name = "_".join([dataset_id.replace("/", "_"), split])
            output_path = f"{output_name}.json"

        with open(output_path, "w") as wf:
            json.dump({
                "answers_unique": answers_unique,
                "generated_texts_unique": generated_texts_unique,
                "scores": scores,
                "average": sum(scores) / len(scores)
            }, wf)
        return average_scores
    accelerator.wait_for_everyone()
    return None


def main():
    parser = argparse.ArgumentParser(description='Calculate TEDS with a HuggingFace Dataset')
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
        '--dataset_id',
        type=str,
        required=True
    )

    parser.add_argument(
        '--output_path',
        type=str,
        default=None
    )

    parser.add_argument(
        '--split',
        type=str,
        default="validation"
    )

    parser.add_argument(
        '--subsampling',
        type=float,
        default=1.0
    )

    args = parser.parse_args()
    evaluate(
        args.model_id_or_path,
        args.dataset_id,
        peft_model=args.peft_model,
        split=args.split,
        subsampling=args.subsampling,
        output_path=args.output_path
    )


if __name__ == "__main__":
    main()
