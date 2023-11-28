import logging
from typing import Dict, Iterable, Optional, Union

import numpy
import torch
from torch.nn import Module
from torch.utils.data import Dataset
from tqdm import tqdm

from datasets import get_data_loader
from evaluator.lipschitz_constant import bound_lipschitz_constant
from utils.line_formatter import LineFormatter
from utils.statistics import Statistics

from .lr_scheduler import PartialLRScheduler
from .metrics import Accuracy, Metric
from .optimizer import PartialOptimizer


class Train:
    eval_metrics: Dict[str, Union[Module, Metric]]
    metrics: Dict[str, numpy.ndarray]

    def __init__(self,
                 model: Module,
                 trainset: Dataset,
                 batch_size: int,
                 epochs: int,
                 loss: Module,
                 device: Union[str, torch.device],
                 optimizer: PartialOptimizer,
                 lr_scheduler: Optional[PartialLRScheduler] = None,
                 valset: Optional[Dataset] = None,
                 eval_metrics: Optional[Dict[str, Metric]] = None,
                 num_workers: int = 2,
                 logger=logging,
                 ):
        self.logger = logger
        self.epochs = epochs

        # Model:
        self.device = device
        self.model = model.to(self.device)

        # Data:
        self.input_shape = trainset[0][0].shape
        self.batch_size = batch_size
        train_loader, val_loader = get_data_loader(
            trainset, valset, batch_size, num_workers, self.logger.info
        )
        self.train_loader = train_loader
        self.val_loader = val_loader

        # Optimizer:
        self.optimizer = optimizer(params=self.model.parameters())
        if lr_scheduler is not None:
            self.lr_scheduler = lr_scheduler(self.optimizer)
        else:
            self.lr_scheduler = None

        # Metric:
        self.loss_fn = loss
        self.init_metrics(loss, eval_metrics)

        self.line_formatter = LineFormatter()

    def init_metrics(self, loss: Module, eval_metrics: Union[None, Dict[str, Metric]]):
        self.eval_metrics = dict(loss=loss)
        self.eval_metrics.update(dict(accuracy=Accuracy()))
        if eval_metrics is not None:
            self.eval_metrics.update(eval_metrics)

        metrics_aggs = dict(epoch='none')
        for name, metric in self.eval_metrics.items():
            metrics_aggs.update({'train_'+name: metric.aggregation})
            metrics_aggs.update({'val_'+name: metric.aggregation})

        self.metrics = dict(
            zip(metrics_aggs.keys(), [numpy.ndarray([])]*len(metrics_aggs)))
        self.statistics = Statistics(metrics_aggs)
        self.logger.info('Statistics to be computed: ')
        self.logger.info(str(self.statistics))

    def update_weights(self, inputs: torch.Tensor, labels: torch.Tensor):
        r"""
        Single batch updating of the weights through the given loss function
        """
        out = self.model(inputs)
        loss = self.loss_fn(out, labels)
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        # Detaching
        loss = loss.detach()
        out = out.detach()
        return loss, out

    @torch.no_grad()
    def evaluate_metrics(self, out: torch.Tensor, labels: torch.Tensor, split: str):
        for name, metric_fn in self.eval_metrics.items():
            metric = tensor2numpy(metric_fn(out, labels))
            self.update_metrics(**{split+'_'+name: metric})
        return

    def train_step(self, epoch: int):  # TODO add into train_module
        self.model.train()
        train_loop = tqdm(self.train_loader,
                          desc=f'Train [{epoch}/{self.epochs}]')
        for (inputs, labels) in train_loop:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            loss, out = self.update_weights(inputs, labels)
            # Evaluate the metrics
            self.evaluate_metrics(out, labels, 'train')

            # update progress bar
            train_accuracy = self.metrics['train_accuracy'].mean()
            train_loop.set_postfix(Loss=tensor2numpy(loss),
                                   Accuracy=train_accuracy)

    @torch.no_grad()
    @torch.nn.utils.parametrize.cached()
    def validation_step(self, epoch: int):
        self.model.eval()
        val_loop = tqdm(self.val_loader,
                        desc=f'Validation[{epoch}/{self.epochs}]')
        for (inputs, labels) in val_loop:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            out = self.model(inputs)
            loss = self.loss_fn(out, labels)

            # Evaluate the metrics
            self.evaluate_metrics(out, labels, 'val')

            # update progress bar
            val_accuracy = self.metrics['val_accuracy'].mean()
            val_loop.set_postfix(Loss=tensor2numpy(loss),
                                 Accuracy=val_accuracy)

    def run(self, root_path: Optional[str] = None, save_freq: int = 10, save_state_dict=True, **addon_print):
        for epoch in range(1, 1+self.epochs):
            self.train_step(epoch)
            # Validate if valset exists
            # self.validation_step(epoch) if self.valset is not None else None
            if self.val_loader is not None:
                self.validation_step(epoch)

            # apply scheduler if scheduler exists
            # self.lr_scheduler.step() if hasattr(self, 'lr_scheduler') else None
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

            # update statistics
            self.update_metrics(epoch=[epoch])
            self.statistics.update(**self.metrics)
            self.reset_metrics()

            self.print_stats(**addon_print)
            if root_path is not None:
                if epoch % save_freq == 0 or epoch == self.epochs:
                    self.logger.info('Saving Checkpoint...')
                    self.save(root_path, save_state_dict)
                    self.logger.info('Done.')
        self.model.eval()
        # model_input_size = self.trainset[0][0].shape[-1]
        model_input_size = self.input_shape[-1]
        ls_bound = bound_lipschitz_constant(
            self.model, model_input_size, 1000)
        self.logger.info(f'Lipschitz constant: {tensor2numpy(ls_bound)}')

    def update_metrics(self, **kwargs):
        for key, val in kwargs.items():
            self.metrics[key] = numpy.append(self.metrics[key], val)

    def reset_metrics(self):
        self.metrics = dict(
            zip(self.metrics.keys(), [numpy.ndarray([])]*len(self.metrics)))

    def print_stats(self, **kwargs):
        r"""
        Print aggregate statistics of the last epoch
        """
        logs = self.statistics.get_last()
        # if hasattr(self, 'lr_scheduler'):
        #     learnig_rate = self.lr_scheduler.get_last_lr()
        #     logs.update({'learning_rate': learnig_rate})
        if self.lr_scheduler is not None:
            learning_rate = self.lr_scheduler.get_last_lr()
            logs.update({'learning_rate': learning_rate})
        logs = {**logs, **kwargs}
        self.logger.info(self.line_formatter.create_line(logs))

    def save(self, path: str, save_state_dict: bool, **addon_save):
        if save_state_dict:
            self.model.eval()
            model_state_dict = self.model.state_dict()
            torch.save(model_state_dict, path+'/model_state_dict.pth')
        # Saving statistics and plots
        self.statistics.save(path+'/training_statistics.csv')
        self.statistics.save_plot(
            path+'/training_statistics.png', subplots=True)


def tensor2numpy(tensor: torch.Tensor):
    return tensor.detach().cpu().numpy()
