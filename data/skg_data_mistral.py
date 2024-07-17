import hydra
from datasets import DatasetDict, concatenate_datasets, load_dataset
from dotenv import load_dotenv
from hydra.core.config_store import ConfigStore

from llamafactory.utils import get_env


cs = ConfigStore.instance()
system_prompt = (
    "Below is an instruction that describes a task, paired with an input that provides further context. "
    "Write a response that appropriately completes the request.\n\n"
)
prompt_template = (
    "Below is an instruction that describes a task, paired with an input that provides further context. "
    "Write a response that appropriately completes the request.\n\n"
    "### Instruction:\n{formatted_instruction}\n\n### Response:\n"
)
instruction_template = "\n\n{instruction}\n\n{input}\n\n{question}"


@hydra.main(
    version_base=None,
    config_path="../configs/generation",
    config_name="config",
)
def main(cfg):
    load_dotenv()
    dataset = load_dataset("TIGER-Lab/SKGInstruct-skg-only", split="train")
    DATA_PATH = get_env("DATA_PATH")

    def format_input_for_generation(x):
        prompt = x["input"]
        label = x["label"]
        task_name = x["task_name"]

        if task_name == "fetaqa":
            splitted_prompt = prompt.split("data table:")
            instruction = splitted_prompt[0]
            try:
                table, question = splitted_prompt[1].split("question:")
            except ValueError:
                table = splitted_prompt[1]
                question = ""
                print("No question found")
            no_cond = ""
        elif task_name == "dart":
            splitted_prompt = prompt.split(":")
            instruction_and_item = splitted_prompt[0]
            instruction = instruction_and_item.split(". ")[0]
            instruction = instruction + "."
            table = prompt[len(instruction) + 1 :]
            question = ""
            no_cond = ""
        elif task_name == "totto":
            instruction = prompt.split("<")[0]
            table = prompt[len(instruction) :]
            question = ""
            no_cond = ""
        else:
            raise ValueError(f"Task {task_name} not recognized.")
        question = question.strip()
        instruction = instruction.strip()
        table = table.strip()
        instruction_only = instruction_template.format(
            instruction=instruction, input=table, question=question
        )
        chat = prompt_template.format(
            instruction=instruction, input=table, question=question
        )

        return {
            "prompt_input": chat,
            "prompt_no_input": no_cond,
            "instruction": instruction_only,
            "real": label,
            "task_name": task_name,
            "system": system_prompt,
        }

    data2text_task_names = ["fetaqa", "dart", "totto"]
    dataset_data2text = dataset.filter(lambda x: x["task_name"] in data2text_task_names)

    dataset_data2text.save_to_disk(str(DATA_PATH / "data" / cfg.data.path))
    dataset_for_generation = dataset_data2text.map(
        format_input_for_generation, num_proc=8
    )
    dataset_for_generation = dataset_for_generation.select_columns(
        [
            "instruction",
            "prompt_input",
            "prompt_no_input",
            "real",
            "task_name",
            "system",
        ]
    )
    if cfg.data.max_samples is not None:
        list_dataset = []
        for task_name in data2text_task_names:
            task_data = dataset_for_generation.filter(
                lambda x: x["task_name"] == task_name
            )
            if hasattr(task_data, "seed"):
                seed = task_data.seed
            else:
                seed = 0
            task_data = task_data.shuffle(seed=seed)
            if isinstance(cfg.data.max_samples, float):
                max_samples = int(len(task_data) * cfg.data.max_samples)
            list_dataset.append(task_data.select(range(max_samples)))
        dataset_for_generation = concatenate_datasets(list_dataset)

    dataset_for_generation = DatasetDict({"train": dataset_for_generation})
    dataset_for_generation.save_to_disk(str(DATA_PATH / cfg.data.path))
    for data, name in zip(list_dataset, data2text_task_names):
        print(f"Task: {name}")
        for i in range(2):
            print(data[i])
            print("--------------------------")


if __name__ == "__main__":
    main()
