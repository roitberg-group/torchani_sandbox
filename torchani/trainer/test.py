import config
import builder
import trainer
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
config_file = 'test.ini'

trainer_config = config.TrainerConfig(config_file)
# print(trainer_config.inputs)

build = builder.Builder(trainer_config, device)

data = trainer.Data(build)
train = trainer.Trainer(build, data)
train.train()
