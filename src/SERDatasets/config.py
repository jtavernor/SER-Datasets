import os
import yaml as pyyaml

class Config:
    def __new__(cls):
        if not hasattr(cls, 'instance'):
            cls.instance = super(Config, cls).__new__(cls)
            dir_path = os.path.dirname(os.path.realpath(__file__))
            run_location = os.getcwd()
            config_path = os.path.join(run_location, 'data_config.yaml')
            if not os.path.exists(config_path):
                with open(os.path.join(dir_path, 'data_config.yaml'), 'r') as config_file:
                    conf = pyyaml.safe_load(config_file)
                with open(config_path, 'w') as config_file:
                    config_file.write(pyyaml.dump(conf))
                raise IOError('Data config file did not exist. Created in current working directory, please confirm config is correct and re-run.')
            with open(config_path, 'r') as config_file:
                cls.instance.config_file = pyyaml.safe_load(config_file)
        return cls.instance.config_file # Returns a shared dictionary