"""
File contains code for training DNN for fcn dataset classification
Trained model is saved in the specified model

Created on : July 17, 2018
Modified on: July 15, 2019
Author: Harshal
"""

import tensorflow as tf
import numpy as np
from random import shuffle
import os
import shutil
import cv2

GPU_ID = 2
input_list_train = 'train_input_shuffled.txt'
label_list_train = 'train_target_shuffled.txt'
input_list_val = 'val_input_shuffled.txt'
label_list_val = 'val_target_shuffled.txt'

if __name__ == "__main__":
    import input_util
    import model

    BATCH_SIZE = 16
    NO_OF_EPOCHS = 1000
    INITIAL_LEARNING_RATE = [5e-4]
    DECAY_STEP = 1000
    DECAY_RATE = 0.1
    LAMBDA = [0.01]
    HEIGHT = 512
    WIDTH = 512

    input_images, label_images = input_util.read_input_file(input_list_train, label_list_train)
    TRAIN_DATASET_SIZE = len(input_images)
    image_in = np.zeros((TRAIN_DATASET_SIZE, HEIGHT, WIDTH, 3), np.uint8)
    label_in = np.zeros((TRAIN_DATASET_SIZE, HEIGHT, WIDTH), np.uint8)
    for i in range(TRAIN_DATASET_SIZE):
        image_in[i] = cv2.imread(input_images[i])
        label_in[i] = cv2.imread(label_images[i], cv2.IMREAD_GRAYSCALE)
    
    mean = image_in.mean(axis = (0, 1, 2)).astype('float32')
    variance = image_in.var(axis = (0, 1, 2)).astype('float32')

    input_images_val, label_images_val = input_util.read_input_file(input_list_val, label_list_val)
    DATASET_SIZE_VAL = len(input_images_val)
    image_in_val = np.zeros((DATASET_SIZE_VAL, HEIGHT, WIDTH, 3), np.uint8)
    label_in_val = np.zeros((DATASET_SIZE_VAL, HEIGHT, WIDTH), np.uint8)
    for i in range(DATASET_SIZE_VAL):
        image_in_val[i] = cv2.imread(input_images_val[i])
        label_in_val[i] = cv2.imread(label_images_val[i], cv2.IMREAD_GRAYSCALE)
    
    accuracy_value = np.zeros((len(LAMBDA), len(INITIAL_LEARNING_RATE)))

    for index_x, lmbd in enumerate(LAMBDA):
        for index_y, in_lr in enumerate(INITIAL_LEARNING_RATE):
            
#             if os.path.exists('./trained_model'):
#                 shutil.rmtree('./trained_model')
            os.makedirs('./trained_model', exist_ok = True)
            np.savetxt('./trained_model/mean.txt', mean)
            np.savetxt('./trained_model/variance.txt', variance)
            
            tf.reset_default_graph()
            
            phase = tf.placeholder(tf.bool)
            image = tf.placeholder(tf.uint8)
            label = tf.placeholder(tf.int32)
            train_dataset = input_util.train_dataset(image, label, BATCH_SIZE, 1)
            train_iterator = train_dataset.make_initializable_iterator()
            val_dataset = input_util.test_dataset(image, label, DATASET_SIZE_VAL)
            val_iterator = val_dataset.make_initializable_iterator()
            
            string_handle = tf.placeholder(tf.string, shape = [])
            dataset_iterator = tf.data.Iterator.from_string_handle(string_handle, train_dataset.output_types, train_dataset.output_shapes)
            data = dataset_iterator.get_next()
            image_queue = data["features"]
            label_queue = data["label"]

            step = tf.train.get_or_create_global_step()
            learning_rate = tf.train.exponential_decay(in_lr, step, DECAY_STEP, DECAY_RATE, staircase = True)
            # learning_rate = tf.constant(in_lr)

            with tf.device('/gpu:{}'.format(GPU_ID)):
                logits = model.FCN2(image_queue, mean, variance, phase)
                loss, train_step = model.train(logits, label_queue, learning_rate, lmbd, step, tf.trainable_variables())
                accuracy = model.old_evaluate(logits, label_queue)

            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            config.allow_soft_placement = True
            # config.log_device_placement = True
            session_args = {
                'checkpoint_dir': './trained_model',
                'save_checkpoint_steps': 300,
                'config': config
            }
            with tf.train.MonitoredTrainingSession(**session_args) as sess:
                train_iterator_string = sess.run(train_iterator.string_handle()) 
                val_iterator_string = sess.run(val_iterator.string_handle()) 

                count = 1
                for epoch in range(NO_OF_EPOCHS):
                    l = list(range(1))
                    shuffle(l)
                    for i in l:
                        sess.run(train_iterator.initializer, feed_dict = {image: image_in, label: label_in})
                        sess.run(val_iterator.initializer, feed_dict = {image: image_in_val, label: label_in_val})

                        while True:
                            try:
                                loss_value, _, lr_value, accuracy_value[index_x, index_y] = sess.run([loss, train_step, learning_rate, accuracy], \
                                        feed_dict = {string_handle: train_iterator_string, phase: True})
                                if count % 20 == 0:
                                    val_loss_value, val_accuracy_value = sess.run([loss, accuracy], feed_dict = {string_handle: val_iterator_string, phase: False})
                                    print("Step: %6d,\tEpoch: %4d,\tLearning Rate: %e,\tTrain Loss: %8.4f,\tTrain Accuracy: %0.4f,\tVal Loss: %8.4f,\tVal Accuracy: %0.4f" %
                                          (count, epoch, lr_value, loss_value, accuracy_value[index_x, index_y], val_loss_value, val_accuracy_value))
                                count += 1
                            except tf.errors.OutOfRangeError:
                                break

                sess.run(val_iterator.initializer, feed_dict = {image: image_in_val, label: label_in_val})
                accuracy_value[index_x, index_y] = sess.run(accuracy, feed_dict = {string_handle: val_iterator_string, phase: False})
                print("Accuracy: ", accuracy_value[index_x, index_y])
                
                sess.run(train_iterator.initializer, feed_dict = {image: image_in, label: label_in})
                accuracy_value[index_x, index_y] = sess.run(accuracy, feed_dict = {string_handle: train_iterator_string, phase: False})
                print("Accuracy: ", accuracy_value[index_x, index_y])

    print(accuracy_value)
