import shutil
import warnings
from dataclasses import dataclass
from typing import Optional

import idr_torch
import torch
import torch.distributed as dist
from datasets import concatenate_datasets, load_from_disk
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import GenerationConfig, PreTrainedTokenizerBase

from llamafactory.generation.decoding import (
    MixtureDecoder,
)
from llamafactory.utils import (
    SavePathFormat,
    get_dataset,
    get_dtype,
    get_model_and_tokenizer,
)


def load_data_distributed(config, batch_size, data_collator, rank, world_size):
    dataset = get_dataset(config.data.path)[config.data.split]

    if config.data.get("max_samples", None) is not None:
        warnings.warn(
            "max_samples is set, only using a subset of the data for evaluation. No shuffling, use for debugging only."
        )
        if config.data.get("select_per_task", False):
            unique_tasks = set(dataset["task_name"])
            dataset = concatenate_datasets(
                [
                    dataset.filter(lambda x: x["task_name"] == task).select(
                        range(config.data.max_samples)
                    )
                    for task in unique_tasks
                ]
            )
        else:
            dataset = dataset.select(range(config.data.max_samples))

    chunk_dataset = dataset.shard(num_shards=world_size, index=rank)
    print(f"rank {rank} has {len(chunk_dataset)} samples")

    data_loader = DataLoader(
        dataset=chunk_dataset,
        batch_size=batch_size,
        drop_last=False,
        shuffle=False,
        collate_fn=data_collator,
    )

    return data_loader


def prepare_generation_inputs(model, model_type, batch):
    if model_type != "standalone" and model_type != "critic":
        if "layer" in model_type:
            generation_inputs = {
                "input_ids": batch[0]["input_ids"].to(model.device),
                "attention_mask": batch[0]["attention_mask"].to(model.device),
                "weak_inputs": {
                    "input_ids": batch[1]["input_ids"].to(model.device),
                    "attention_mask": batch[1]["attention_mask"].to(model.device),
                },
            }
        else:
            generation_inputs = {
                "input_ids": batch[0]["input_ids"].to(model.device),
                "attention_mask": batch[0]["attention_mask"].to(model.device),
                "weak_inputs": [
                    {
                        "input_ids": weak_batch["input_ids"].to(model.device),
                        "attention_mask": weak_batch["attention_mask"].to(model.device),
                    }
                    for weak_batch in batch[1]
                ],
            }

    else:
        generation_inputs = {
            "input_ids": batch["input_ids"].to(model.device),
            "attention_mask": batch["attention_mask"].to(model.device),
        }
    return generation_inputs


def run_generation(model, config, data_loader):
    generation_config = GenerationConfig(
        return_dict_in_generate=True, output_scores=True, **config.generation
    )
    sequences = []
    dtype = get_dtype(config.eval.dtype)
    sequences_wo_instuctions = []
    print("Running generation")
    for batch in tqdm(data_loader):
        generation_inputs = prepare_generation_inputs(
            model, model_type=config.model_type, batch=batch
        )

        with torch.no_grad():
            with torch.cuda.amp.autocast(dtype=dtype):
                generation_out = model.generate(
                    generation_config=generation_config,
                    output_scores=True,
                    return_dict_in_generate=True,
                    **generation_inputs,
                )

        input_len = generation_inputs["input_ids"].shape[1]

        sequences_wo_instuctions += generation_out.sequences.cpu()[
            :, input_len:
        ].tolist()
        sequences += generation_out.sequences.cpu().tolist()

    # truncate sequences to remove padding from generation
    for i in range(len(sequences_wo_instuctions)):
        if config.generation.eos_token_id in sequences_wo_instuctions[i]:
            stop_idx = sequences_wo_instuctions[i].index(config.generation.eos_token_id)
        else:
            stop_idx = len(sequences_wo_instuctions[i])
        sequences_wo_instuctions[i] = sequences_wo_instuctions[i][: stop_idx + 1]

    return {
        "sequences_instructions": sequences,
        "sequences": sequences_wo_instuctions,
    }


def run_generation_distributed(config, init_dist=True):
    rank = idr_torch.rank
    world_size = idr_torch.world_size

    print(
        f"{rank} process", "ngpus:", torch.cuda.device_count(), "world_size", world_size
    )

    data_collator, model, tokenizer = load_data_collator_model_tokenizer(config)
    if rank == 0:
        save_format = SavePathFormat(config)
        out_path = save_format.get_save_path()
        out_path_object = [out_path]
        tmp_path = save_format.get_tmp_path()
        tmp_path_object = [tmp_path]
    else:
        out_path_object = [None]
        tmp_path_object = [None]
    if init_dist:
        dist.init_process_group(
            backend="mpi", init_method="env://", world_size=world_size, rank=rank
        )
    dist.broadcast_object_list(out_path_object, src=0)
    dist.broadcast_object_list(tmp_path_object, src=0)
    out_path = out_path_object[0]
    tmp_path = tmp_path_object[0]
    print("Saving to:", out_path)

    model.eval()
    data_loader = load_data_distributed(
        config,
        batch_size=config.eval.batch_size,
        data_collator=data_collator,
        rank=rank,
        world_size=world_size,
    )

    generation_dict = run_generation(model, config, data_loader)
    dataset = data_loader.dataset
    dataset = dataset.add_column("generated_ids", generation_dict["sequences"])
    dataset = dataset.add_column(
        "generated_ids_instructions", generation_dict["sequences_instructions"]
    )
    dataset = dataset.map(
        lambda x: {
            "prediction": tokenizer.batch_decode(
                x["generated_ids"],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True,
            )
        },
        batched=True,
        batch_size=100,
    )
    dataset = dataset.map(
        lambda x: {
            "prediction_instructions": tokenizer.batch_decode(
                x["generated_ids_instructions"],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True,
            )
        },
        batched=True,
        batch_size=100,
    )

    print(f"finished rank {rank}")
    if rank == 0:
        if tmp_path.exists():
            shutil.rmtree(tmp_path)
        dataset.save_to_disk(str(tmp_path / f"predictions_and_scores_{rank}"))
        print("saved rank 0")
    for i in range(1, world_size):
        dist.barrier()
        if rank == i:
            dataset.save_to_disk(str(tmp_path / f"predictions_and_scores_{rank}"))
            print(f"saved rank {rank}")
    dist.barrier()
    if rank == 0:
        list_datasets = []
        for i in range(world_size):
            list_datasets.append(
                load_from_disk(
                    str(tmp_path / f"predictions_and_scores_{i}"),
                )
            )
        dataset_with_results = concatenate_datasets(list_datasets, axis=0)
        dataset_with_results.save_to_disk(str(out_path / "predictions_and_scores"))
        dataset_with_results.to_csv(str(out_path / "predictions_and_scores.csv"))
        print(f"Saved to {out_path / 'predictions_and_scores'}")
        OmegaConf.save(config=config, f=out_path / "config.yaml")
        print_gen_out(config, dataset_with_results)
        return dataset_with_results


def run_generation_not_distributed(config):
    data_collator, model, tokenizer = load_data_collator_model_tokenizer(config)

    save_format = SavePathFormat(config)
    out_path = save_format.get_save_path()

    print("Saving to:", out_path)

    model.eval()
    data_loader = load_data_distributed(
        config,
        batch_size=config.eval.batch_size,
        data_collator=data_collator,
        rank=0,
        world_size=1,
    )

    generation_dict = run_generation(model, config, data_loader)
    dataset = data_loader.dataset
    dataset = dataset.add_column("generated_ids", generation_dict["sequences"])
    dataset = dataset.add_column(
        "generated_ids_instructions", generation_dict["sequences_instructions"]
    )
    dataset = dataset.map(
        lambda x: {
            "prediction": tokenizer.batch_decode(
                x["generated_ids"],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True,
            )
        },
        batched=True,
        batch_size=100,
    )
    dataset = dataset.map(
        lambda x: {
            "prediction_instructions": tokenizer.batch_decode(
                x["generated_ids_instructions"],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True,
            )
        },
        batched=True,
        batch_size=100,
    )

    print_gen_out(config, dataset)
    return dataset


def run_generation_interactive(config):
    data_collator, model, tokenizer = load_data_collator_model_tokenizer(config)

    save_format = SavePathFormat(config)
    out_path = save_format.get_save_path()

    print("Saving to:", out_path)

    model.eval()
    data_loader = load_data_distributed(
        config,
        batch_size=config.eval.batch_size,
        data_collator=data_collator,
        rank=0,
        world_size=1,
    )

    def gen():
        generation_dict = run_generation(model, config, data_loader)
        dataset = data_loader.dataset
        dataset = dataset.add_column("generated_ids", generation_dict["sequences"])
        dataset = dataset.add_column(
            "generated_ids_instructions", generation_dict["sequences_instructions"]
        )
        dataset = dataset.map(
            lambda x: {
                "prediction": tokenizer.batch_decode(
                    x["generated_ids"],
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=True,
                )
            },
            batched=True,
            batch_size=100,
        )
        dataset = dataset.map(
            lambda x: {
                "prediction_instructions": tokenizer.batch_decode(
                    x["generated_ids_instructions"],
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=True,
                )
            },
            batched=True,
            batch_size=100,
        )

        print_gen_out(config, dataset)

    import ipdb

    ipdb.set_trace()


def print_gen_out(config, dataset):
    task_names = set(dataset["task_name"])
    for task in task_names:
        task_dataset = dataset.filter(lambda x: x["task_name"] == task)
        max_samples = config.data.get("max_samples", None)
        if max_samples is None:
            max_samples = len(task_dataset)
        print(f"\n\nTASK: {task}\n\n")
        for i in range(min(5, max_samples, len(dataset))):
            sample = task_dataset[i]

            print("====================================")
            print("Prompt:\n", sample["prompt_input"])
            print("Real:\n", sample["real"])
            print("------------------------------------")
            print("Gen:\n", sample["prediction"])


def load_data_collator_model_tokenizer(config, load_model=True):
    model, tokenizer = get_model_and_tokenizer(config, load_model=load_model)

    data_collator = DataCollatorForLM(
        tokenizer=tokenizer,
        max_length=config.model.max_length,
        pad_to_multiple_of=8,
        return_tensors="pt",
    )
    if config.model_type != "standalone":
        assert (
            config.noise is not None
        ), "Noise model must be provided in guided decoding system."
        model = load_expert_guided_decoder(config, model, load_model=load_model)
        noise_data_collator = DataCollatorForLM(
            tokenizer=tokenizer,
            max_length=config.noise.max_length,
            pad_to_multiple_of=8,
            return_tensors="pt",
            with_input=False,
        )

        data_collator = ExpertGuidedDataCollator(data_collator, noise_data_collator)
    model.eval()
    return data_collator, model, tokenizer


def load_expert_guided_decoder(config, main_model, load_model=True):

    main_config = config.model
    noise_config = config.noise

    assert (
        noise_config is not None
    ), "You need to supply either a noise model or a weak model."

    if load_model:
        if noise_config.model_path == main_config.model_path:
            print("Using same backbone for noise model")
            noise_model = main_model
        else:
            cfg_copy = config.copy()
            cfg_copy.model = noise_config
            noise_model, _ = get_model_and_tokenizer(cfg_copy, load_model=True)
            main_vocab_size = main_model.config.vocab_size
            noise_vocab_size = noise_model.config.vocab_size
            if noise_vocab_size != main_vocab_size:
                if main_vocab_size > noise_vocab_size:
                    print(
                        f"Extending noise model vocab from {noise_vocab_size} to {main_vocab_size}"
                    )
                    noise_model.resize_token_embeddings(main_vocab_size)
                    config.generation.suppress_tokens = config.generation.get(
                        "suppress_tokens", []
                    ) + list(range(noise_vocab_size, main_vocab_size))

    else:
        noise_model = None

    if not load_model:
        return None
    else:
        model_type = config.model_type
        if model_type == "mixture":
            model = MixtureDecoder(
                model=main_model,
                unconditional_model=noise_model,
                mixture_alpha=config.generation.mixture_alpha,
                mixture_mode=config.generation.mixture_mode,
                n_untouched_logits=config.generation.mixture_n_untouched,
            )
            return model
        elif model_type == "context-aware":
            model = MixtureDecoder(
                model=main_model,
                unconditional_model=noise_model,
                mixture_alpha=config.generation.context_aware_alpha,
                mixture_mode="cad",
            )
            return model

        else:
            raise ValueError(f"Model type {model_type} not recognized")


@dataclass
class DataCollatorForLM:
    tokenizer: PreTrainedTokenizerBase
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    return_tensors: str = "pt"
    label_pad_token_id = -100
    with_input: bool = True

    def __call__(self, batch, return_tensors=None):
        if self.with_input:
            formatted_batch = [x["prompt_input"] for x in batch]
        else:
            formatted_batch = [x["prompt_no_input"] for x in batch]
        if return_tensors is None:
            return_tensors = self.return_tensors

        features = self.tokenizer(
            formatted_batch,
            add_special_tokens=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors=return_tensors,
            padding=True,
            pad_to_multiple_of=self.pad_to_multiple_of,
        )

        return features


class ExpertGuidedDataCollator:
    def __init__(self, main_data_collator, *args):
        self.main_data_collator = main_data_collator
        self.weak_data_collators = args

    def __call__(self, batch):
        main_input = self.main_data_collator(batch)
        weak_inputs = [
            weak_collator(batch) for weak_collator in self.weak_data_collators
        ]
        return main_input, weak_inputs
