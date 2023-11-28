import argparse
from typing import Protocol, Dict

import datasets
import train_model
import measure_memories
import measure_batch_times

from models import get_all_model_layer_combinations


class ExecuteTask(Protocol):
    def __call__(self,
                 idx: int,
                 model_name: str,
                 layer_name: str,
                 results_file_name: str,
                 dataset_cls: datasets.DatasetClass):
        ...


DATASET_CLASSES = {
    "cifar10": datasets.CIFAR10,
    "cifar100": datasets.CIFAR100,
    "tiny-image-net": datasets.TinyImageNet,
}

TASKS: Dict[str, ExecuteTask] = {
    "train-model": train_model.define_and_train_model,
    "measure-memory": measure_memories.measure_memories,
    "measure-batch-times": measure_batch_times.measure_batch_times,
    "print-job-id-assignment": lambda *args: print_job_id_assignment(),
}


def index_to_model_layer_name(idx: int):
    all_combinations = get_all_model_layer_combinations(
        use_tiny_image_net=False,
        logging=print,
    )
    model_name, layer_name = all_combinations[idx]
    print(f"Chosen combination {idx}: "
          f"model {model_name} and layer {layer_name}.\n")
    return model_name, layer_name


def parse_command_line_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--job-id", type=int, default=0)
    parser.add_argument("--results-file-name", type=str, default=None)
    ds_keys = DATASET_CLASSES.keys()
    parser.add_argument("--dataset", choices=ds_keys, default="cifar10")
    parser.add_argument("--task", choices=TASKS.keys(), default="train-model")

    cmd_args = parser.parse_args()
    return cmd_args


def print_job_id_assignment():
    all_combinations = get_all_model_layer_combinations(
        use_tiny_image_net=False,
        logging=print,
    )
    print(f"Found {len(all_combinations)} combinations: ")
    for idx, (model_name, layer_name) in enumerate(all_combinations):
        print(f"{idx: 3d}: {model_name:^12} & {layer_name:^15}")


def main():
    cmd_args = parse_command_line_arguments()
    print(cmd_args)

    model_name, layer_name = index_to_model_layer_name(cmd_args.job_id)
    dataset_cls = DATASET_CLASSES[cmd_args.dataset]
    run_task = TASKS[cmd_args.task]

    run_task(
        cmd_args.job_id,
        model_name,
        layer_name,
        cmd_args.results_file_name,
        dataset_cls
    )


if __name__ == "__main__":
    main()
