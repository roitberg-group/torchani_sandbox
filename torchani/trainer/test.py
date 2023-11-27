import config
import builder

device = 'cpu'
config_file = 'simple.ini'

trainer_config = config.TrainerConfig(config_file)
#print(trainer_config.inputs)

build = builder.ModelBuilder(trainer_config, device)
nn, model = build.standard_model()
modules = build.modules
