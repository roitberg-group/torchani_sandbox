import config
import builder
import trainer

device = 'cpu'
config_file = 'simple.ini'

trainer_config = config.TrainerConfig(config_file)
#print(trainer_config.inputs)

build = builder.Builder(trainer_config, device)
nn, model = build.standard_model()
modules = build.modules

data = trainer.Data(build)
train = trainer.Trainer(build, data)
train.learning_rate_scheduler()
