import yaml
from .tagged_types import TaggedScalar, TaggedSequence, TaggedMapping
from yaml.dumper import SafeDumper


class TaggedDumper(SafeDumper):
    @staticmethod
    def scalar_representer(dumper: yaml.Dumper, data):
        return dumper.represent_scalar(data.tag, data.value, style=data.style)

    @staticmethod
    def sequence_representer(dumper: yaml.Dumper, data):
        return dumper.represent_sequence(data.tag, data.value)

    @staticmethod
    def mapping_representer(dumper: yaml.Dumper, data):
        return dumper.represent_mapping(data.tag, data.value)


TaggedDumper.add_representer(TaggedScalar, TaggedDumper.scalar_representer)
TaggedDumper.add_representer(TaggedSequence, TaggedDumper.sequence_representer)
TaggedDumper.add_representer(TaggedMapping, TaggedDumper.mapping_representer)
