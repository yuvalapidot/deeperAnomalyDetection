from keras.layers import Input, Dense, Convolution1D, AveragePooling1D, UpSampling1D, Activation
from keras.models import Model
import numpy as np

from base import BaseRepresentation
import memory_slice_generator as generation

from theano.sandbox.cuda.dnn import *
print(dnn_available())
print(dnn_available.msg)


input_output_size = 4096
nb_filter = 64
filter_length = 256
# filter_lengths = [524288, 4096, 1024, 256, 64]
activation = Activation('softplus')

input_arr = Input(shape=(1, input_output_size,), name="input")
x = Convolution1D(nb_filter=nb_filter, filter_length=filter_length, activation='softplus',
                  input_dim=input_output_size)(input_arr)
encoded = AveragePooling1D(pool_length=2)(x)

x = Convolution1D(nb_filter=nb_filter, filter_length=filter_length, activation='softplus')(encoded)
x = UpSampling1D(2)(x)
decoded = Convolution1D(nb_filter=1, filter_length=filter_length, activation='softplus')(x)

autoencoder = Model(input_arr, decoded)
autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')

train_generator = generation.MemorySliceGenerator(["D:\DeepFeaturesExperiment\Dumps\TRAIN"], input_output_size,
                                                  base=BaseRepresentation.byte)
test_generator = generation.MemorySliceGenerator(["D:\DeepFeaturesExperiment\Dumps\TEST"], input_output_size,
                                                 base=BaseRepresentation.byte)

autoencoder.fit_generator(generator=train_generator.generate_memory_slices(), samples_per_epoch=1024, nb_epoch=200,
                          verbose=1, validation_data=test_generator.generate_memory_slices(), nb_val_samples=100,
                          pickle_safe=True)
