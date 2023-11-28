import time
import unittest

import datasets
from evaluator.batch_times import (
    evaluate_seconds_per_batch,
    # evaluate_training_batch_times,
    # evaluate_validation_batch_times,
)


class TestBatchTimes(unittest.TestCase):
    def test_with_dummy_model(self):
        batch_times = evaluate_seconds_per_batch(
            evaluation_function=lambda x: time.sleep(1.),
            loader=[(1, 1) for _ in range(3)],
            nrof_batches=2,
        )
        assert len(batch_times) == 2
        assert 1. <= batch_times[0] <= 1.1
        assert 1. <= batch_times[1] <= 1.1

    def test_iterator_cycling(self):
        batch_times = evaluate_seconds_per_batch(
            evaluation_function=lambda x: x,
            loader=[(1, 1) for _ in range(3)],
            nrof_batches=10
        )
        assert len(batch_times) == 10

    def test_batch_times_on_cifar(self):
        train_dataset = datasets.CIFAR10(train=True)
        train_loader, _ = datasets.get_data_loader(
            train_dataset,
            train_dataset,
            batch_size=256,
            num_workers=4,
            log=print,
        )
        batch_times = evaluate_seconds_per_batch(
            evaluation_function=lambda x: x,
            loader=train_loader,
            nrof_batches=10,
        )
        assert len(batch_times) == 10
        assert all(isinstance(t, float) for t in batch_times)
        assert all(0. <= t < 0.1 for t in batch_times)

    # def test_plot_batch_time_distribution(self):
    #     import matplotlib.pyplot as plt
    #
    #     from models.simplified_conv_net import (
    #         SimplifiedConvNet, DEFAULT_MODELS
    #     )
    #     from models.layers import AOLConv2d, MaxMin
    #
    #     nrof_batches = 3
    #     model = SimplifiedConvNet(get_conv=AOLConv2d,
    #                               get_activation=MaxMin,
    #                               **DEFAULT_MODELS["ConvNetM"])
    #
    #     train_loader, test_loader
    #     = datasets.get_cifar_10_train_test_loaders()
    #
    #     train_batch_times = evaluate_training_batch_times(
    #         model, train_loader, nrof_batches=nrof_batches, device="cpu")
    #     print(train_batch_times)
    #     plt.plot(train_batch_times, label="training")
    #
    #     val_batch_times = evaluate_validation_batch_times(
    #         model, test_loader,
    #         nrof_batches=nrof_batches, cached=False, device="cpu")
    #     print(val_batch_times)
    #     plt.plot(val_batch_times, label="validation")
    #
    #     cached_val_batch_times = evaluate_validation_batch_times(
    #         model, test_loader,
    #         nrof_batches=nrof_batches, device="cpu")
    #     print(cached_val_batch_times)
    #     plt.plot(cached_val_batch_times, label="cached_validation")
    #
    #     plt.legend()
    #     plt.gca().set_ylim(bottom=0)
    #     plt.title("Batch times")
    #     plt.xlabel("Batch number")
    #     plt.ylabel("Time (s)")
    #     plt.show()
