import numpy as np
from generator import build_generator
from discriminator import (
    build_discriminator_net, 
    build_discriminator_supervised,
    build_discriminator_unsupervised
)

from gan import build_gan
from config import *
from keras.optimizers import Adam
from dataset import Dataset
from keras.utils import to_categorical

discriminator_net = build_discriminator_net(img_shape)
discriminator_supervised = build_discriminator_supervised(discriminator_net)
discriminator_supervised.compile(
    loss='categorical_crossentropy',
    metrics=['accuracy'],
    optimizer=Adam()
)

discriminator_unsupervised = build_discriminator_unsupervised(discriminator_net)
discriminator_unsupervised.compile(
    loss='binary_crossentropy',
    optimizer=Adam()
)

generator = build_generator(z_dim)
discriminator_unsupervised.trainable = False 
gan = build_gan(generator, discriminator_unsupervised)
gan.compile(loss='binary_crossentropy', optimizer=Adam())

supervised_losses = []
iteration_checkpoints = []

def train(iterations, batch_size, sample_intervals):
    real = np.ones((batch_size, 1))
    fake = np.zeros((batch_size, 1))

    for iteration in range(iterations):
        imgs, labels = dataset.batch_labeled(batch_size)
        labels = to_categorical(labels, num_classes=num_classes)
        
        imgs_unlabeled = dataset.batch_unlabeled(batch_size)

        z = np.random.normal(0, 1, (batch_size, z_dim))
        gen_imgs = generator.predict(z)

        d_loss_supervised, accuracy = discriminator_supervised.train_on_batch(imgs, labels)
        d_loss_real = discriminator_unsupervised.train_on_batch(imgs_unlabeled, real)
        d_loss_fake = discriminator_unsupervised.train_on_batch(gen_imgs, fake)

        d_loss_unsupervised = 0.5 * np.add(d_loss_real, d_loss_fake)
        
        z = np.random.normal(0, 1, (batch_size, z_dim))
        gen_imgs = generator.predict(z)

        g_loss = gan.train_on_batch(z, np.ones((batch_size, 1)))

        if (iteration + 1) % sample_interval == 0:
            supervised_losses.append(d_loss_supervised)
            iteration_checkpoints.append(iteration + 1)

            print(
                "%d [D loss supervised: %.4f, acc.: %.2f%%] [D loss " +
                " unsupervised: %.4f] [G loss: %f]"
                % (iteration + 1, d_loss_supervised, 100*accuracy,
                (d_loss_unsupervised, g_loss))
            )

if __name__ == '__main__':
    dataset = Dataset(num_labeled)
    train(iterations, batch_size, sample_interval)

    # test 
    x, y = dataset.test_set()
    y = to_categorical(y, num_classes=num_classes)

    _, accuracy = discriminator_supervised.evaluate(x, y)
    print("Test Accuracy: %.2f%%" % (100 * accuracy))