import torch
from torch import Tensor
from torch.nn import Module
from torch.utils.data import Dataset, DataLoader
import logging
from .lipschitz_constant import bound_lipschitz_constant
from trainer.metrics import Margin
from torch.nn.utils.parametrize import cached

import pandas as pd
import numpy
SQRT2 = 2**0.5


class EvaluateRobustAcc:
    model: Module
    valset: Dataset
    logger: logging.Logger

    def __init__(self, model: Module,
                 valset: Dataset,
                 loss: Module,
                 device: torch.device,
                 logger=logging,
                 ):
        self.logger = logger
        self.init_model(model, device)
        self.init_data(valset)
        self.init_metrics()
        self.init_lip()
        self.get_margin = Margin()

    def init_model(self, model: Module, device: torch.device):
        self.device = device
        self.model = model.to(device)
        self.model.eval()

    def init_data(self, valset: Dataset):
        self.valset = valset
        self.test_loader = DataLoader(valset, batch_size=100, shuffle=False)

    def init_metrics(self):
        self.metrics = {
            'label': [],
            'pred': [],
            'margin': [],
            'margin_lip': [],
        }

    def init_lip(self):
        self.logger.info('Estimating the lipschitz constant...')
        self.lip = bound_lipschitz_constant(
            self.model, nrof_iterations=10000)
        self.logger.info(f'Lipschitz constant of the model: {self.lip}')

    def update_metrics(self, results: dict):
        for key, value in results.items():
            new_values = self.tensor2numpy(value)
            old_values = self.metrics[key]
            self.metrics[key] = numpy.append(old_values, new_values)

    @torch.no_grad()
    def evaluate_batch(self, x: Tensor, labels: Tensor):
        out = self.model(x)
        pred = out.argmax(-1).long()
        margin = self.get_margin(out, labels) / SQRT2
        margin_lip = margin/self.lip
        # todo use foolbox to evaluate the attack
        results = dict(
            label=labels.long(),
            pred=pred,
            margin=margin,
            margin_lip=margin_lip
        )
        return results

    @cached()
    def run(self, save_path: str):
        for idx, (inputs, labels) in enumerate(self.test_loader, 1):
            self.logger.info(f'Evaluating batch {idx}')
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            cur_results = self.evaluate_batch(inputs, labels)
            self.update_metrics(cur_results)
            if idx % 10 == 0 or idx == len(self.test_loader):
                self.save(save_path)
        self.logger.info('Done.')

    def save(self, save_path: str):
        # Create a setting.csv to save the metrics
        self.logger.info(f'Saving metrics')
        df = pd.DataFrame(self.metrics)
        df.to_csv(save_path+'metrics.csv', index=False)

    @staticmethod
    def tensor2numpy(tensor: torch.Tensor):
        return tensor.detach().cpu().numpy()
