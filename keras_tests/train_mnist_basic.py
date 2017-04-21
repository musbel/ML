import keras
from keras.layers.core import Flatten
from keras.layers.core import Dense
from keras.layers.core import Dropout
from keras.layers import Input
from keras.models import Model
from keras.regularizers import *

'''
Train a simple deep neural network model on the mnist dataset (without convolutions)

Generated using deepcognition.ai
TODO: Modify to use in standalone mode
'''

def get_model():
	aliases = {}
	Input_1 = Input(shape=(28, 28, 1), name='Input_1')
	Flatten_1 = Flatten(name='Flatten_1')(Input_1)
	Dense_1 = Dense(name='Dense_1', activation='relu', output_dim=512)(Flatten_1)
	Dropout_1 = Dropout(name='Dropout_1',p= 0.3)(Dense_1)
	Dense_2 = Dense(name='Dense_2', activation='relu', output_dim=512)(Dropout_1)
	Dropout_2 = Dropout(name='Dropout_2',p= 0.3)(Dense_2)
	Dense_3 = Dense(name='Dense_3', activation='softmax', output_dim=10)(Dropout_2)

	model = Model([Input_1], [Dense_3])
	return aliases, model


from keras.optimizers import *

def get_optimizer():
	return Adadelta()

def is_custom_loss_function():
	return False

def get_loss_function():
	return 'categorical_crossentropy'

def get_batch_size():
	return 32

def get_num_epoch():
	return 10

def get_data_config():
	return '{"mapping": {"Image": {"type": "Image", "options": {"height_shift_range": 0, "Width": 28, "Height": 28, "Normalization": false, "Resize": false, "shear_range": 0, "Scaling": 1, "width_shift_range": 0, "horizontal_flip": false, "vertical_flip": false, "Augmentation": false, "pretrained": "None", "rotation_range": 0}, "port": "InputPort0"}, "Digit Label": {"type": "Categorical", "options": {}, "port": "OutputPort0"}}, "kfold": 1, "shuffle": false, "samples": {"training": 42000, "test": 14000, "validation": 14000, "split": 1}, "numPorts": 1, "datasetLoadOption": "batch", "dataset": {"type": "public", "name": "mnist", "samples": 70000}}'