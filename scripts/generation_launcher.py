from pathlib import Path

import hydra
import idr_torch
from dotenv import load_dotenv

from llamafactory.generation.generation import (
    run_generation_distributed,
    run_generation_not_distributed,
)
from llamafactory.utils import SavePathFormat, get_env


def get_output_dir(cfg):
    save_path_format = SavePathFormat(cfg)
    model_folder_name = save_path_format.get_model_folder_name()
    data = cfg.data.path
    output_dir = Path(data)
    output_dir = output_dir / model_folder_name
    return output_dir


def run_generation_and_evaluation(cfg, init_dist=True):

    load_dotenv()
    if cfg.do_gen:
        print("Launching generation.")
        rank = idr_torch.rank
        world_size = idr_torch.world_size

        if world_size > 1:
            gen_dataset = run_generation_distributed(cfg, init_dist=init_dist)
        else:
            gen_dataset = run_generation_not_distributed(cfg)

        if cfg.loser_gen:
            output_dir = get_output_dir(cfg)
            RESULT_PATH = get_env("RESULT_PATH")
            output_dir = RESULT_PATH / output_dir

            output_dir.mkdir(parents=True, exist_ok=True)
            if rank == 0:

                gen_dataset = gen_dataset.rename_column("prediction", "generated")
                gen_dataset = gen_dataset.select_columns(
                    [
                        "generated",
                        "real",
                        "prompt_input",
                        "prompt_no_input",
                        "instruction",
                        "task_name",
                        "system",
                    ]
                )

                gen_dataset.save_to_disk(str(output_dir))
                print(f"Saved to {output_dir}")


@hydra.main(
    version_base=None, config_path="../configs/generation", config_name="config"
)
def main(cfg):
    load_dotenv()
    run_generation_and_evaluation(cfg)


if __name__ == "__main__":
    main()
