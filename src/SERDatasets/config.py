import os
import yaml as pyyaml

class Config:
    def __new__(cls):
        if not hasattr(cls, 'instance'):
            cls.instance = super(Config, cls).__new__(cls)
            dir_path = os.path.dirname(os.path.realpath(__file__))
            run_location = os.getcwd()
            config_path = os.path.join(dir_path, 'data_config.yaml')
            with open(config_path, 'r') as config_file:
                cls.instance.config_file = pyyaml.safe_load(config_file)
        return cls.instance.config_file # Returns a shared dictionary