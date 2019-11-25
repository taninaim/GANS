import matplotlib.pyplot as plt 
import numpy as np 

from keras.datasets import mnist 
from keras.layers import (
    Activation, BatchNormalization, Concatenate, Dense,
    Embedding, Flatten, Input, Multiply, Reshape,
    Conv2D, Conv2DTranspose
)
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Model, Sequential
from keras.optimizers import Adam 

img_rows = 28
img_cols = 28
channels = 1

img_shape = (img_rows, img_cols, channels)
z_dim = 100
num_classes = 10

def build_generator(z_dim):
    model = Sequential()

    model.add(Dense(256 * 7 * 7, input_dim=z_dim))
    model.add(Reshape((7, 7, 256)))

    model.add(Conv2DTranspose(128, 
                              kernel_size=3,
                              strides=2,
                              padding='same'))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.01))

    model.add(Conv2DTranspose(64, 
                              kernel_size=3,
                              strides=1,
                              padding='same'))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.01))

    model.add(Conv2DTranspose(1, 
                              kernel_size=3,
                              strides=2,
                              padding='same'))

    model.add(Activation('tanh'))

    return model 

def build_cgan_generator(z_dim):
    z = Input(shape=(z_dim,))
    label = Input(shape=(1,), dtype='int32')
    label_embedding = Embedding(num_classes, z_dim, input_length=1)(label)
    label_embedding = Flatten()(label_embedding)
    joined_representation = Multiply()([z, label_embedding])
    generator = build_generator(z_dim)
    conditioned_img = generator(joined_representation)
    return Model([z, label], conditioned_img)

def build_discriminator(img_shape):
    model = Sequential()

    model.add(
        Conv2D(64,
               kernel_size=3,
               strides=2,
               input_shape=(img_shape[0], img_shape[1], img_shape[2] + 1),
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

    model.add(Flatten())
    model.add(Dense(1, activation='sigmoid'))
    
    return model

def build_cgan_discriminator(img_shape):
    img = Input(shape=img_shape)
    label = Input(shape=(1,), dtype='int32')
    label_embedding = Embedding(num_classes,
                                np.prod(img_shape),
                                input_length=1)(label)

    label_embedding = Flatten()(label_embedding)
    label_embedding = Reshape(img_shape)(label_embedding)
    concatenated = Concatenate(axis=-1)([img, label_embedding])
    discriminator = build_discriminator(img_shape)
    classification = discriminator(concatenated)
    return Model([img, label], classification)

def build_cgan(generator, discriminator):
    z = Input(shape=(z_dim,))
    label = Input(shape=(1,))
    img = generator([z, label])
    classification = discriminator([img, label])
    model = Model([z, label], classification)
    return model 

discriminator = build_cgan_discriminator(img_shape)
discriminator.compile(
    loss='binary_crossentropy',
    optimizer=Adam(),
    metrics=['accuracy']
)

generator = build_cgan_generator(z_dim)

discriminator.trainable = False 

cgan = build_cgan(generator, discriminator)

cgan.compile(loss='binary_crossentropy', optimizer=Adam())

