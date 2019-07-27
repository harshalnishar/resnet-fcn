"""
File contains code for inferring input image with trained model for fcn dataset classification

Created on: July 18, 2018
Author: Harshal
"""

import tensorflow as tf
import numpy as np
import cv2


if __name__ == "__main__":
    import input_util
    import model

    HEIGHT = 512
    WIDTH = 512

    mean = np.loadtxt('./trained_model/mean.txt', dtype = np.float32)
    variance = np.loadtxt('./trained_model/variance.txt', dtype = np.float32)

    input_list_val = 'val_input_shuffled.txt'
    label_list_val = 'val_target_shuffled.txt'
    input_images_val, label_images_val = input_util.read_input_file(input_list_val, label_list_val)
    DATASET_SIZE_VAL = len(input_images_val)

    image_in_val = np.zeros((DATASET_SIZE_VAL, HEIGHT, WIDTH, 3), np.uint8)
    label_in_val = np.zeros((DATASET_SIZE_VAL, HEIGHT, WIDTH), np.uint8)
    for i in range(DATASET_SIZE_VAL):
        image_in_val[i] = cv2.imread(input_images_val[i])
        label_in_val[i] = cv2.imread(label_images_val[i], cv2.IMREAD_GRAYSCALE)

    image = tf.placeholder(tf.uint8)
    label = tf.placeholder(tf.int32)

    dataset = input_util.test_dataset(image, label, DATASET_SIZE_VAL)
    dataset_iterator = dataset.make_initializable_iterator()
    data = dataset_iterator.get_next()
    image_queue = data["features"]
    label_queue = data["label"]
    
    with tf.device('/gpu:2'):
        logits = model.FCN2(image_queue, mean, variance, False)
        prediction, probability = model.predict(logits)
        _, accuracy = model.evaluate(logits, label_queue)

    saver_handle = tf.train.Saver()

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    with tf.Session(config = config) as sess:
        saver_handle.restore(sess, tf.train.latest_checkpoint('./trained_model'))
        sess.run(tf.local_variables_initializer())

        sess.run(dataset_iterator.initializer, feed_dict = {image: image_in_val, label: label_in_val})
        image_out, label_out, prediction_out, accuracy_val = sess.run([image_queue, label_queue, prediction, accuracy])
        #prediction_out, probability_out, actual = sess.run([prediction, probability, label_queue])
        #print("Prediction: %d with Probability: %f\nActual: %d" % (prediction_out, probability_out, actual))
        print(prediction_out)
        print("Accuracy: %f" % (accuracy_val))

    prediction_out[prediction_out == 1] = 255
    label_out[label_out > 0] = 255
    for i in range(DATASET_SIZE_VAL):
        name0 = 'output/image_%d.png' % i
        name1 = 'output/predi_%d.png' % i
        name2 = 'output/label_%d.png' % i
        cv2.imwrite(name0, image_out[i])
        cv2.imwrite(name1, prediction_out[i])
        cv2.imwrite(name2, label_out[i])

