"""
File contains routine for importing and manipulating input data

Author: Harshal
Created on : July 6, 2018
Modified on: July 15, 2019
"""

import tensorflow as tf
import numpy as np
import cv2


def read_input_file(input_list, label_list):
    """
    fuction to read input image and ground truth image from file list
    :param input_list: path string to input image file list
    :param label_list: path string to label image file list

    """
    file = open(input_list)
    input_images = file.readlines()
    input_images = [f.strip()[:-2] for f in input_images]
    file.close()
    file = open(label_list)
    label_images = file.readlines()
    label_images = [f.strip()[:-2] for f in label_images]
    file.close()
    return (input_images, label_images)

def train_dataset(image, label, batch_size, epochs):
    """
    function to create and return train dataset iterator
    :param image: input image tensor
    :param label: input label tensor
    :param batch_size: batch size for dataset batching
    :param epochs: number of epochs (dataset repetition)
    :return dataset_iterator: initialisable iterator over dataset
    """
    dataset_object = tf.data.Dataset.from_tensor_slices({"features": image, "label": label})
    dataset_object = dataset_object.shuffle(10000)
    dataset_object = dataset_object.repeat(epochs)
    dataset_object = dataset_object.batch(batch_size)
    return dataset_object

def test_dataset(image, label, batch_size):
    """
    Function to create and return test dataset iterator
    :param image: input image tensor
    :param label: input label tensor
    :param batch_size: batch size for dataset batching
    :param epochs: number of epochs (dataset repetition)
    :return dataset_iterator: initialisable iterator over dataset
    """
    dataset_object = tf.data.Dataset.from_tensor_slices({"features": image, "label": label})
    dataset_object = dataset_object.batch(batch_size)
    return dataset_object

if __name__ == '__main__':

    BATCH_SIZE = 2
    NO_OF_EPOCHS = 10
    WIDTH = 512
    HEIGHT = 512

    input_list = 'train_input_shuffled.txt'
    label_list = 'train_target_shuffled.txt'
    input_images, label_images = read_input_file(input_list, label_list)

    image_in = np.zeros((BATCH_SIZE, HEIGHT, WIDTH, 3), np.uint8)
    label_in = np.zeros((BATCH_SIZE, HEIGHT, WIDTH), np.uint8)
    for i in range(BATCH_SIZE):
        image_in[i] = cv2.imread(input_images[i])
        label_in[i] = cv2.imread(label_images[i], cv2.IMREAD_GRAYSCALE)

    image = tf.placeholder(tf.uint8)
    label = tf.placeholder(tf.int32)

    dataset_iterator = train_dataset(image, label, BATCH_SIZE, NO_OF_EPOCHS)
    with tf.Session() as sess:
        print(input_list)
        sess.run(dataset_iterator.initializer, feed_dict = {image: image_in, label: label_in})
        data = sess.run(dataset_iterator.get_next())
        print(data)

