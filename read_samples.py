#!/usr/bin/python3

from mnist import MNIST
from scipy.misc import toimage
import PIL

print('Getting MNIST data...')
mndata = MNIST('./mnist')

print('Retrieving training data...')
images, labels = mndata.load_training()

samples = 10

for index in range(samples):
    imageMatrix = []
    i = 0

    for y in range(28):
        line = []

        for x in range(28):
            line.append(images[index][i])
            i += 1

        imageMatrix.append(line)

    image = toimage(imageMatrix).resize([256, 256], PIL.Image.ANTIALIAS)

    image.show()

    input('(' + str(index) + ') It\'s a ' + str(labels[index]) + '. Next...')