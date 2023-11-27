from configparser import ConfigParser


class TrainerConfig:

    def __init__(self, config_file, device):
        self.config = self.load_config(config_file)
        self.device = device
        self.model_config = self.parse_config()

    def load_config(self, config_file):
        config = ConfigParser(allow_no_value=True, inline_comment_prefixes="#")
        config.read(config_file)
        return config

    def parse_config(self):
        gv = self.config['Global']
        tv = self.config['Trainer']

        model_config = {}

        for key, value in gv.items():
            self._convert(key, value, model_config)

        for key, value in tv.items():
            self._convert(key, value, model_config)

        assert not (model_config['netlike1x'] and model_config['netlike2x']), "In configuration file, netlike1x and netlike2x cannot both be True"

        if model_config['netlike1x']:
            model_config['constants']  = '../resources/ani-1x_8x/rHCNO-5.2R_16-3.5A_a4-8.params'
            model_config['elements'] = ['H', 'C', 'N', 'O']
        elif model_config['netlike2x']:
            model_config['constants'] = '../resources/ani-2x_8x/rHCNOSFCl-5.1R_16-3.5A_a8-4.params'
            model_config['elements'] = ['H', 'C', 'N', 'O', 'S', 'F', 'Cl']
        else: 
            assert 'constants' in model_config and model_config['constants'] is not None, "constants not specified in configuration file."
            assert 'elements' in model_config and model_config['elements'] is not None, "elements not specified in configuration file."
        return model_config

    def _convert(self, key, value, model_config):
        try:
            model_config[key] = eval(value)

        except (NameError, SyntaxError):
            model_config[key] = value
