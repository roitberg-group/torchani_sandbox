import config
import builder

device = 'cpu'
config_file = 'simple.ini'

trainer_config = config.TrainerConfig(config_file, device)
#print(trainer_config.inputs)

build = builder.ModelBuilder(trainer_config)
build.forward()
