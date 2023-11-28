
import logging

import torch

from models import get_model
from trainer import Train, metrics


NROF_EPOCHS = 10
PEAK_LR = 1e-3


def define_and_train_model(idx, model_name, layer_name, _, dataset_cls):
    model = get_model(model_name, layer_name)
    train_dataset = dataset_cls(train=True)
    train_model(model, train_dataset)


def train_model(model, train_dataset):
    loss = metrics.LipCrossEntropyLoss(
        margin=2 * 2**0.5 * 36/255,
        temperature=1/4,
    )

    def get_optimizer(params):
        return torch.optim.SGD(
            params,  # model_parameters,
            lr=0.,
            weight_decay=10 ** -5,
            momentum=0.9,
        )

    def get_lr_scheduler(optimizer):
        return torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=PEAK_LR,
            epochs=NROF_EPOCHS,
            steps_per_epoch=1,
        )

    eval_metrics = {
        "accuracy": metrics.Accuracy(),
        "loss": loss,
        "cra36": metrics.CRA(36/255),
        "cra72": metrics.CRA(72/255),
        "cra108": metrics.CRA(108/255),
        "margin": metrics.SignedMargin(),
        "batch_var": metrics.BatchVariance(),
        "throughput": metrics.Throughput(),
    }
    logging.basicConfig(level=logging.DEBUG)

    trainer = Train(
        model,
        train_dataset,
        batch_size=256,
        epochs=NROF_EPOCHS,
        loss=loss,
        device="cuda" if torch.cuda.is_available() else "cpu",
        optimizer=get_optimizer,
        lr_scheduler=get_lr_scheduler,
        valset=None,
        eval_metrics=eval_metrics,
        num_workers=4,
        logger=logging,
    )

    trainer.run()



