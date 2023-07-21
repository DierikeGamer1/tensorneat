import os
import warnings
import configparser

import numpy as np


class Configer:

    @classmethod
    def __load_default_config(cls):
        par_dir = os.path.dirname(os.path.abspath(__file__))
        default_config_path = os.path.join(par_dir, "default_config.ini")
        return cls.__load_config(default_config_path)

    @classmethod
    def __load_config(cls, config_path):
        c = configparser.ConfigParser()
        c.read(config_path)
        config = {}

        for section in c.sections():
            for key, value in c.items(section):
                config[key] = eval(value)

        return config

    @classmethod
    def __check_redundant_config(cls, default_config, config):
        for key in config:
            if key not in default_config:
                warnings.warn(f"Redundant config: {key} in config!")

    @classmethod
    def __complete_config(cls, default_config, config):
        for key in default_config:
            if key not in config:
                config[key] = default_config[key]

    @classmethod
    def load_config(cls, config_path=None):
        default_config = cls.__load_default_config()
        if config_path is None:
            config = {}
        elif not os.path.exists(config_path):
            warnings.warn(f"config file {config_path} not exist!")
            config = {}
        else:
            config = cls.__load_config(config_path)

        cls.__check_redundant_config(default_config, config)
        cls.__complete_config(default_config, config)

        cls.refactor_activation(config)
        cls.refactor_aggregation(config)

        config['input_idx'] = np.arange(config['num_inputs'])
        config['output_idx'] = np.arange(config['num_inputs'], config['num_inputs'] + config['num_outputs'])

        return config

    @classmethod
    def refactor_activation(cls, config):
        config['activation_default'] = 0
        config['activation_options'] = np.arange(len(config['activation_option_names']))

    @classmethod
    def refactor_aggregation(cls, config):
        config['aggregation_default'] = 0
        config['aggregation_options'] = np.arange(len(config['aggregation_option_names']))
