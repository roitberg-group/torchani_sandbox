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

        return model_config 

    def _convert(self, key, value, model_config):
        try:
            model_config[key] = eval(value)

        except(NameError, SyntaxError):
            model_config[key] = value
