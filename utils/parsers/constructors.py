import random
from typing import Any, Type

import yaml
from yaml.loader import SafeLoader

import datasets
import models
from trainer import lr_scheduler, metrics, optimizer

from .tagged_types import TaggedMapping, TaggedScalar, TaggedSequence


class DeterministicLoader(SafeLoader):
    @staticmethod
    def model_constructor(loader: yaml.SafeLoader, node: yaml.nodes.MappingNode) -> models.layers.Module:
        kwargs = loader.construct_mapping(node, deep=True)
        classname = kwargs['name']
        kwargs.pop('name')
        return getattr(models, classname)(**kwargs)

    @staticmethod
    def layer_constructor(loader: yaml.SafeLoader, node: yaml.nodes.ScalarNode) -> Type[models.layers.Module]:
        return getattr(models.layers, loader.construct_scalar(node))

    @staticmethod
    def dataset_constructor(loader: yaml.SafeLoader, node: yaml.nodes.MappingNode) -> datasets.Dataset:
        kwargs = loader.construct_mapping(node)
        classname = kwargs['name']
        kwargs.pop('name')
        return getattr(datasets, classname)(**kwargs)

    @staticmethod
    def optimizer_constructor(loader: yaml.SafeLoader, node: yaml.nodes.MappingNode) -> Type[optimizer.Optimizer]:
        kwargs = loader.construct_mapping(node)
        classname = kwargs['name']
        kwargs.pop('name')
        return getattr(optimizer, classname)(**kwargs)

    @staticmethod
    def scheduler_constructor(loader: yaml.SafeLoader, node: yaml.nodes.MappingNode) -> Type[lr_scheduler.LRScheduler]:
        kwargs = loader.construct_mapping(node)
        classname = kwargs['name']
        kwargs.pop('name')
        return getattr(lr_scheduler, classname)(**kwargs)

    @staticmethod
    def metric_constructor(loader: yaml.SafeLoader, node: yaml.nodes.MappingNode) -> Type[metrics.Metric]:
        kwargs = loader.construct_mapping(node)
        classname = kwargs['name']
        kwargs.pop('name')
        return getattr(metrics, classname)(**kwargs)


class RandomLoader(SafeLoader):
    # Random constructors
    @staticmethod
    def randint_constructor(loader: yaml.SafeLoader, node: yaml.nodes.SequenceNode) -> int:
        args = loader.construct_sequence(node)
        float_args = map(int, args)
        return random.randint(*float_args)

    @staticmethod
    def randfloat_constructor(loader: yaml.SafeLoader, node: yaml.nodes.SequenceNode) -> float:
        args = loader.construct_sequence(node, deep=True)
        int_args = map(float, args)
        return random.uniform(*int_args)

    @staticmethod
    def choice_constructor(loader: yaml.SafeLoader, node: yaml.nodes.SequenceNode) -> Any:
        choiches = loader.construct_sequence(node)
        return random.choice(choiches)

    @staticmethod
    def random_log10_constructor(loader: yaml.SafeLoader, node: yaml.nodes.SequenceNode) -> float:
        args = loader.construct_sequence(node, deep=True)
        int_args = map(float, args)
        return 10 ** random.uniform(*int_args)

    @staticmethod
    def tagged_constructor(loader: yaml.Loader, tag, node):
        if isinstance(node, yaml.ScalarNode):
            return TaggedScalar(tag, loader.construct_scalar(node))
        if isinstance(node, yaml.SequenceNode):
            return TaggedSequence(tag, loader.construct_sequence(node, deep=True))
        if isinstance(node, yaml.MappingNode):
            return TaggedMapping(tag, loader.construct_mapping(node, deep=True))
        else:
            raise NotImplementedError('Node: ' + str(type(node)))


RandomLoader.add_constructor("!randint", RandomLoader.randint_constructor)
RandomLoader.add_constructor("!randfloat", RandomLoader.randfloat_constructor)
RandomLoader.add_constructor(
    "!randlog10", RandomLoader.random_log10_constructor)
RandomLoader.add_constructor("!choice", RandomLoader.choice_constructor)
RandomLoader.add_multi_constructor("", RandomLoader.tagged_constructor)


DeterministicLoader.add_constructor(
    "!model", DeterministicLoader.model_constructor)
DeterministicLoader.add_constructor(
    "!layer", DeterministicLoader.layer_constructor)
DeterministicLoader.add_constructor(
    "!dataset", DeterministicLoader.dataset_constructor)
DeterministicLoader.add_constructor(
    "!optimizer", DeterministicLoader.optimizer_constructor)
DeterministicLoader.add_constructor(
    "!scheduler", DeterministicLoader.scheduler_constructor)
DeterministicLoader.add_constructor(
    "!metric", DeterministicLoader.metric_constructor)
