from keras.layers import *
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Model, Sequential

def build_gan(generator, discriminator):
    model = Sequential()
    model.add(generator)
    model.add(discriminator)
    return model

    