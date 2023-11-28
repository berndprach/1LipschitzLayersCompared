import yaml
from .constructors import RandomLoader, DeterministicLoader
from .dumpers import TaggedDumper
from .tagged_types import Tagged


class SettingParser:
    # Required since the deterministic setting has to be stored but not in dict
    __slots__ = ['__dict__', '__sampled_setting__']

    def __init__(self, setting_path: str) -> None:
        self.__sampled_setting__ = self.sample_setting(setting_path)

    @staticmethod
    def sample_setting(fname: str):
        """ Samples values such as !randint. """
        with open(fname, 'r') as file:
            sampled_setting = yaml.load(file, Loader=RandomLoader)
        return sampled_setting

    @staticmethod
    def dump_settings(sampled):
        dumped_settings = yaml.dump(sampled, Dumper=TaggedDumper,
                                    default_flow_style=False,
                                    allow_unicode=True)
        return dumped_settings

    def load_setting(self):
        """ Loads classes as e.g.
        !model, will be initialized
        !layer, will not be initialized and feed to the model class
        !optimizer, will be partialy initialized
        !lr_scheduler, will be partialy initialized
        !loss, will be initialized """
        dumped_setting = self.dump_settings(self.__sampled_setting__)
        objects = yaml.load(dumped_setting, Loader=DeterministicLoader)
        self.__dict__ = objects
        return self

    def update_setting(self, new_setting: str):
        r'''
        Update the configuration with a string in the yaml setting format.
        e.g
        parser = SettingParser('setting.yaml')
        parser.update_setting(r"
                                device: cuda:0
        "
        )
        '''
        # Safely convert into Tagged nodes
        update_setting = yaml.load(new_setting, Loader=RandomLoader)
        self.__sampled_setting__.update(update_setting)

    def save_setting(self, setting_path: str):
        dumped_setting = self.dump_settings(self.__sampled_setting__)
        with open(setting_path, 'w') as file:
            file.write(dumped_setting)

    def __item__(self, key: str):
        r'''
        Given a key returns the associated value at any level
        '''
        def _finditem(key: str, obj: dict):
            if key in obj.keys():
                value = obj[key]
                if isinstance(value, Tagged):
                    value = value.value
                return value
            for v in obj.values():
                value = v
                if isinstance(v, Tagged):
                    value = v.value
                if isinstance(value, dict):
                    item = _finditem(key, value)
                    if item is not None:
                        return item
        return _finditem(key, self.__sampled_setting__)

    def __repr__(self) -> str:
        dumped_setting = self.dump_settings(self.__sampled_setting__)
        return dumped_setting

    def to_dict(self):
        return vars(self)
