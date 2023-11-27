import config

device = 'cpu'
config_file = 'simple.ini'

trainer_config = config.TrainerConfig(config_file, device)
print(trainer_config.model_config)
