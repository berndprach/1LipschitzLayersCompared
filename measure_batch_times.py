import os

import datasets

from models import get_model
from evaluator.batch_times import evaluate_all_model_time_statistics

SEPERATOR = ";"
NROF_BATCHES = 100  # 250?


def measure_batch_times(idx: int,
                        model_name: str,
                        layer_name: str,
                        results_file_name: str,
                        dataset_cls: datasets.DatasetClass):
    model = get_model(model_name, layer_name)
    if results_file_name is None:
        results_file_name = f"measured_batch_times.csv"
    train_loader, test_loader = datasets.get_train_test_loader(dataset_cls)

    results_folder = os.path.join("data", "evaluations")
    os.makedirs(results_folder, exist_ok=True)
    results_file = os.path.join(results_folder, results_file_name)

    # Draw a line before a new set of evaluations:
    if idx == 0:
        with open(results_file, "a") as f:
            f.write("=" * 30 + "\n")

    exception = None
    exception_str = str(None)
    result_dict = None

    try:
        result_dict = evaluate_all_model_time_statistics(
            model,
            train_loader=train_loader,
            test_loader=test_loader,
            nrof_batches=NROF_BATCHES,
            log=print,
        )

    except Exception as e:
        print(f"Error while evaluating "
              f"model {model_name} and layer {layer_name}!\n "
              f"{e}")

        exception = e
        exception_str = str(e).replace(SEPERATOR, ",")

    with open(results_file, "a") as f:
        sep = SEPERATOR
        f.write(f"{idx}{sep} "
                f"{model_name}{sep} "
                f"{layer_name}{sep} "
                f"{result_dict}{sep} "
                f"{exception_str}\n")

    if exception is not None:
        raise exception
