import os
from datetime import datetime
from pathlib import Path

import torch
from datasets import load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer


def get_env(env_variable):
    return Path(os.environ[env_variable])


def get_dataset(data_path):
    DATA_PATH = get_env("DATA_PATH")
    return load_from_disk(str(DATA_PATH / data_path))


def get_model_and_tokenizer(cfg, load_model=True):
    CHECKPOINT_PATH = get_env("CHECKPOINT_PATH")

    model_path = CHECKPOINT_PATH / cfg.model.model_path
    tokenizer_path = model_path

    dtype = get_dtype(cfg.eval.dtype)

    attn_implementation = cfg.model.get("attn_implementation", "flash_attention_2")

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, legacy=False)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    tokenizer.truncation_side = "left"

    model_class = AutoModelForCausalLM

    model_kwargs = {
        "device_map": "auto",
        "attn_implementation": attn_implementation,
    }

    model = model_class.from_pretrained(
        str(model_path),
        pad_token_id=tokenizer.pad_token_id,
        torch_dtype=dtype,
        **model_kwargs,
    )

    return model, tokenizer


class SavePathFormat:
    """
    Format the path for saving and loading the results.
    """

    def __init__(self, config):
        self.config = config

    def get_tmp_path(self):
        TMP_PATH = get_env("TMPDIR")
        print(f"TMPDIR: {TMP_PATH}")
        job_id = os.environ.get("SLURM_JOB_ID", "0")
        current_date = datetime.now().strftime("%m-%d-%H:%M:%S")

        return TMP_PATH / f"{job_id}_{current_date}"

    @staticmethod
    def get_model_name(model_config):
        model_path = model_config.model_path
        return model_path

    def get_task_name(self):
        return self.config.task

    def get_model_folder_name(self):
        folder_name = (
            f"{self.config.model_type}/{self.get_model_name(self.config.model)}/"
        )
        if self.config.model_type == "mixture":
            folder_name = (
                folder_name
                + f"{self.config.generation.mixture_mode}_a{self.config.generation.mixture_alpha}_u{self.config.generation.mixture_n_untouched}"
            )
        if self.config.model_type == "context-aware":
            folder_name = folder_name + f"{self.config.generation.context_aware_alpha}"
        if self.config.model_type == "contrastive":
            folder_name = folder_name + f"{self.config.generation.contrastive_alpha}"
        if self.config.model_type == "pmi":
            folder_name = (
                folder_name
                + f"_l{self.config.generation.pmi_lambda}_t{self.config.generation.pmi_tau}"
            )

        if (
            self.config.model_type != "standalone"
            and self.config.model_type != "critic"
        ):
            noise_model_name = self.get_model_name(self.config.noise)
            folder_name = folder_name + "_" + f"noise{noise_model_name}"

        return folder_name

    def get_path(self):
        RESULT_PATH = get_env("RESULT_PATH")
        split_path = self.config.data.path.split("/")
        if len(split_path) > 1:
            out_path = RESULT_PATH / split_path[-2] / split_path[-1]
        else:
            out_path = RESULT_PATH / split_path[-1]

        folder_name = self.get_model_folder_name()
        return out_path / folder_name

    def get_save_path(self):
        out_path = self.get_path()
        current_time = datetime.now().strftime("%m-%d-%H:%M:%S")
        out_path = out_path / current_time
        out_path.mkdir(exist_ok=True, parents=True)
        return out_path

    def get_results_path(self, date="latest"):
        out_path = self.get_path()

        if date == "latest":
            # Get latest timestamp in folder
            folders = list(os.listdir(out_path))
            timestamps_datetime = [
                datetime.strptime(ts, "%m-%d-%H:%M:%S") for ts in folders
            ]
            latest_folder = max(timestamps_datetime).strftime("%m-%d-%H:%M:%S")
        else:
            latest_folder = date

        out_path = out_path / latest_folder
        return out_path

    def get_generation_results_path(self, date="latest"):
        return self.get_results_path(date) / "predictions"


def get_dtype(type_str):
    if type_str == "bf16":
        return torch.bfloat16
    elif type_str == "fp16":
        return torch.float16
    else:
        return torch.float32
