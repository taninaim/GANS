from keras import backend as K
from keras.layers import *
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Model, Sequential
from config import num_classes

def build_discriminator_net(img_shape):
    model = Sequential()

    model.add(
        Conv2D(32,
               kernel_size=3,
               strides=2,
               input_shape=img_shape,
               padding='same'))
    
    model.add(LeakyReLU(alpha=0.01))

    model.add(
        Conv2D(64,
               kernel_size=3,
               strides=2,
               input_shape=img_shape,
               padding='same'))

    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.01))

    model.add(
        Conv2D(128,
               kernel_size=3,
               strides=2,
               input_shape=img_shape,
               padding='same'))

    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.01))

    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(num_classes))
    return model

def build_discriminator_supervised(discriminator_net):
    model = Sequential()
    model.add(discriminator_net)
    model.add(Activation('softmax'))
    return model

def build_discriminator_unsupervised(discriminator_net):
    model = Sequential()
    model.add(discriminator.net)

    def predict(x):
        prediction = 1.0 - (1.0 / 
            (K.sum(K.exp(x), axis=-1, keepdims=True) + 1.0))
        return prediction
    
    model.add(Lambda(predict))
    return model
