"""
File contains neural network model for fcn dataset classification
with use of residual blocks

Created on: October 26, 2018
Author: Harshal
"""

import tensorflow as tf
import resnet_model
import input_util
import numpy as np
import cv2

def FCN1(image, mean, variance, phase):
    """
    function defining fcn model
    :param image: input image tensor in the flattened form
    :return: model output tensor node
    """

    image = tf.cast(image, tf.float32)
    image_reshape = tf.reshape(image, [-1, 512, 512, 3])

    image_norm = tf.nn.batch_normalization(image_reshape, mean, variance, None, None, 0.0001)

    model = resnet_model.Model(
        resnet_size = 50,
        bottleneck = False,
        num_classes = 2,
        num_filters = 64,
        kernel_size = 7,
        conv_stride = 2,
        first_pool_size = 3,
        first_pool_stride = 2,
        block_sizes = [3, 4, 6, 3],
        block_strides = [1, 2, 2, 2],
        resnet_version = 1,
        data_format = 'channels_last',
    )
    resnet_out = model(image_norm, phase)
    intermediate_out = [v.values()[0] for v in tf.get_default_graph().get_operations() if 'block_layer' in v.name]

    conv1 = tf.layers.conv2d(intermediate_out[3], filters = 512, kernel_size = [1, 1], strides = (1, 1), padding = 'same',
                             activation = tf.nn.relu, name = 'conv1_1x1')
    
    conv2 = tf.layers.conv2d(conv1, filters = 512, kernel_size = [1, 1], strides = (1, 1), padding = 'same',
                             activation = tf.nn.relu, name = 'conv2_1x1')

    up1 = tf.layers.conv2d_transpose(conv2, filters = 128, kernel_size = [8, 8], strides = (4, 4), padding = 'same',
                                     activation = tf.nn.relu, name = 'upsample_1')

    up2 = tf.layers.conv2d_transpose(intermediate_out[2], filters = 128, kernel_size = [4, 4], strides = (2, 2), padding = 'same',
                                     activation = tf.nn.relu, name = 'upsample_2')

    up3 = intermediate_out[1]

    up4 = tf.layers.conv2d_transpose(up3 + up2 + up1, filters = 2, kernel_size = [16, 16], strides = (8, 8), padding = 'same',
                                     activation = None, name = 'upsample_4')

    return up4


def FCN2(image, mean, variance, phase):
    """
    function defining fcn model
    :param image: input image tensor in the flattened form
    :return: model output tensor node
    """

    image = tf.cast(image, tf.float32)
    image_reshape = tf.reshape(image, [-1, 512, 512, 3])

    image_norm = tf.nn.batch_normalization(image_reshape, mean, variance, None, None, 0.0001)

    model = resnet_model.Model(
        resnet_size = 50,
        bottleneck = False,
        num_classes = 2,
        num_filters = 64,
        kernel_size = 7,
        conv_stride = 2,
        first_pool_size = 3,
        first_pool_stride = 2,
        block_sizes = [3, 4, 6, 3],
        block_strides = [1, 2, 2, 2],
        resnet_version = 1,
        data_format = 'channels_last',
    )
    resnet_out = model(image_norm, phase)
    intermediate_out = [v.values()[0] for v in tf.get_default_graph().get_operations() if 'block_layer' in v.name]

    conv1 = tf.layers.conv2d(intermediate_out[3], filters = 512, kernel_size = [1, 1], strides = (1, 1), padding = 'same',
                             activation = tf.nn.relu, name = 'conv1_1x1')
    
    conv2 = tf.layers.conv2d(conv1, filters = 512, kernel_size = [1, 1], strides = (1, 1), padding = 'same',
                             activation = tf.nn.relu, name = 'conv2_1x1')

    up1 = tf.layers.conv2d_transpose(conv2, filters = 64, kernel_size = [16, 16], strides = (8, 8), padding = 'same',
                                     activation = tf.nn.relu, name = 'upsample_1')

    up2 = tf.layers.conv2d_transpose(intermediate_out[2], filters = 64, kernel_size = [8, 8], strides = (4, 4), padding = 'same',
                                     activation = tf.nn.relu, name = 'upsample_2')

    up3 = tf.layers.conv2d_transpose(intermediate_out[1], filters = 64, kernel_size = [4, 4], strides = (2, 2), padding = 'same',
                                     activation = tf.nn.relu, name = 'upsample_3')

    up4 = intermediate_out[0]

    up5 = tf.layers.conv2d_transpose(up4 + up3 + up2 + up1, filters = 2, kernel_size = [8, 8], strides = (4, 4), padding = 'same',
                                     activation = None, name = 'upsample_5')

    return up5


def FCN3(image, mean, variance, phase):
    """
    function defining fcn model
    :param image: input image tensor in the flattened form
    :return: model output tensor node
    """

    image = tf.cast(image, tf.float32)
    image_reshape = tf.reshape(image, [-1, 512, 512, 3])

    image_norm = tf.nn.batch_normalization(image_reshape, mean, variance, None, None, 0.0001)

    model = resnet_model.Model(
        resnet_size = 50,
        bottleneck = False,
        num_classes = 2,
        num_filters = 64,
        kernel_size = 7,
        conv_stride = 2,
        first_pool_size = 3,
        first_pool_stride = 2,
        block_sizes = [3, 4, 6, 3],
        block_strides = [1, 2, 2, 2],
        resnet_version = 1,
        data_format = 'channels_last',
    )
    resnet_out = model(image_norm, phase)
    intermediate_out = [v.values()[0] for v in tf.get_default_graph().get_operations() if 'block_layer' in v.name]

    conv1 = tf.layers.conv2d(intermediate_out[3], filters = 512, kernel_size = [1, 1], strides = (1, 1), padding = 'same',
                             activation = tf.nn.relu, name = 'conv1_1x1')
    
    conv2 = tf.layers.conv2d(conv1, filters = 512, kernel_size = [1, 1], strides = (1, 1), padding = 'same',
                             activation = tf.nn.relu, name = 'conv2_1x1')

    up1 = tf.layers.conv2d_transpose(conv2, filters = 256, kernel_size = [4, 4], strides = (2, 2), padding = 'same',
                                     activation = tf.nn.relu, name = 'upsample_1')

    up2 = tf.layers.conv2d_transpose(up1 + intermediate_out[2], filters = 128, kernel_size = [4, 4], strides = (2, 2), padding = 'same',
                                     activation = tf.nn.relu, name = 'upsample_2')

    up3 = tf.layers.conv2d_transpose(up2 + intermediate_out[1], filters = 64, kernel_size = [4, 4], strides = (2, 2), padding = 'same',
                                     activation = tf.nn.relu, name = 'upsample_3')

    up4 = tf.layers.conv2d_transpose(up3, filters = 2, kernel_size = [8, 8], strides = (4, 4), padding = 'same',
                                     activation = None, name = 'upsample_4')

    return up4


def FCN4(image, mean, variance, phase):
    """
    function defining fcn model
    :param image: input image tensor in the flattened form
    :return: model output tensor node
    """

    image = tf.cast(image, tf.float32)
    image_reshape = tf.reshape(image, [-1, 512, 512, 3])

    image_norm = tf.nn.batch_normalization(image_reshape, mean, variance, None, None, 0.0001)

    model = resnet_model.Model(
        resnet_size = 50,
        bottleneck = False,
        num_classes = 2,
        num_filters = 64,
        kernel_size = 7,
        conv_stride = 2,
        first_pool_size = 3,
        first_pool_stride = 2,
        block_sizes = [3, 4, 6, 3],
        block_strides = [1, 2, 2, 2],
        resnet_version = 1,
        data_format = 'channels_last',
    )
    resnet_out = model(image_norm, phase)
    intermediate_out = [v.values()[0] for v in tf.get_default_graph().get_operations() if 'block_layer' in v.name]

    conv1 = tf.layers.conv2d(intermediate_out[3], filters = 512, kernel_size = [1, 1], strides = (1, 1), padding = 'same',
                             activation = tf.nn.relu, name = 'conv1_1x1')
    
    conv2 = tf.layers.conv2d(conv1, filters = 512, kernel_size = [1, 1], strides = (1, 1), padding = 'same',
                             activation = tf.nn.relu, name = 'conv2_1x1')

    up1 = tf.layers.conv2d_transpose(conv2, filters = 256, kernel_size = [4, 4], strides = (2, 2), padding = 'same',
                                     activation = tf.nn.relu, name = 'upsample_1')

    up2 = tf.layers.conv2d_transpose(up1 + intermediate_out[2], filters = 128, kernel_size = [4, 4], strides = (2, 2), padding = 'same',
                                     activation = tf.nn.relu, name = 'upsample_2')

    up3 = tf.layers.conv2d_transpose(up2 + intermediate_out[1], filters = 64, kernel_size = [4, 4], strides = (2, 2), padding = 'same',
                                     activation = tf.nn.relu, name = 'upsample_3')

    up4 = tf.layers.conv2d_transpose(up3 + intermediate_out[0], filters = 3, kernel_size = [4, 4], strides = (2, 2), padding = 'same',
                                     activation = tf.nn.relu, name = 'upsample_4')

    up5 = tf.layers.conv2d_transpose(up4, filters = 2, kernel_size = [4, 4], strides = (2, 2), padding = 'same',
                                     activation = None, name = 'upsample_5')

    return up5

def predict(logits):
    """
    function outputs the predicted class based on input logits
    :param logits: logits tensor
    :return: returns predicted output with prediction probability
    """
    prediction = tf.cast(tf.argmax(logits, axis = 3), tf.int32)
    probability = tf.reduce_max(tf.nn.softmax(logits), axis = 3)
    return prediction, probability
    # return prediction, probability

def old_evaluate(logits, labels):
    """
    function to evaluate the logits output against labels with normal accuracy
    :param logits: logits tensor
    :param labels: normal (not one hot) labels tensor
    :return: prediction accuracy
    """
    prediction, _ = predict(logits)
    match = tf.equal(labels, prediction)
    accuracy = tf.reduce_mean(tf.cast(match, tf.float32))
    return accuracy

def evaluate(logits, labels):
    """
    function to evaluate the logits output against labels with running sum accuracy
    :param logits: logits tensor
    :param labels: normal (not one hot) labels tensor
    :return: prediction running accuracy
    """
    prediction, _ = predict(logits)
    accuracy, accuracy_op = tf.metrics.accuracy(labels = labels, predictions = prediction)
    # mean_iou, iou_op = tf.metrics.mean_iou(labels = labels, predictions = prediction, num_classes = 2) 
    # return accuracy, accuracy_op, mean_iou, iou_op
    return accuracy, accuracy_op

def train(logits, labels, learning_rate, l2_regularization, step, train_var):
    """
    function to train the dnn model for fcn training set
    :param logits: logits tensor
    :param labels: normal (not one hot) labels tensor
    :param learning_rate: initial learning rate or learning rate function
    :param l2_regularization: regularization factor
    :param step: global training step needed for learning rate function
    :param train_var: list of tensor variables which must be used for minimization
    :return: training loss and training operation
    """
    loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits = logits, labels = labels))
    tf.cast(loss, dtype = tf.float64)
    cost = tf.identity(loss)
    if l2_regularization is not None:
        loss_l2 = tf.reduce_mean([tf.nn.l2_loss(v) for v in tf.trainable_variables() if (('bias' not in v.name) and ('batch_normalization' not in v.name))])
        cost = cost + l2_regularization * loss_l2
    optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate)
    # optimizer = tf.train.GradientDescentOptimizer(learning_rate = learning_rate)
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        optimizer_step = optimizer.minimize(cost, global_step = step, var_list = train_var)
    return loss, optimizer_step


if __name__ == "__main__":
    import input_util

    DATASET_SIZE = 500
    BATCH_SIZE = 16
    NO_OF_EPOCHS = 1
    LEARNING_RATE = 10e-4
    LAMBDA = None #0.05
    HEIGHT = 512
    WIDTH = 512

    image = tf.placeholder(tf.uint8)
    label = tf.placeholder(tf.int32)
    phase = tf.placeholder(tf.bool)

    dataset = input_util.train_dataset(image, label, BATCH_SIZE, NO_OF_EPOCHS)
    dataset_iterator = dataset.make_initializable_iterator()
    data = dataset_iterator.get_next()
    image_queue = data["features"]
    label_queue = data["label"]

    step = tf.train.get_or_create_global_step()
    logits = dnn(image_queue, 0.0, 1.0, phase)
    loss, train_step = train(logits, label_queue, LEARNING_RATE, LAMBDA, step, tf.trainable_variables())
    accuracy = old_evaluate(logits, label_queue)

    input_list = 'train_input_shuffled.txt'
    label_list = 'train_target_shuffled.txt'
    input_images, label_images = input_util.read_input_file(input_list, label_list)
    DATASET_SIZE = len(input_images)
    
    image_in = np.zeros((DATASET_SIZE, HEIGHT, WIDTH, 3), np.uint8)
    label_in = np.zeros((DATASET_SIZE, HEIGHT, WIDTH), np.uint8)
    for i in range(DATASET_SIZE):
        image_in[i] = cv2.imread(input_images[i])
        label_in[i] = cv2.imread(label_images[i], cv2.IMREAD_GRAYSCALE)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        sess.run(dataset_iterator.initializer, feed_dict = {image: image_in, label: label_in})

        count = 1
        while True:
            try:
                loss_value, _, accuracy_value = sess.run([loss, train_step, accuracy], feed_dict = {phase: True})
                if count % 5 == 0:
                    print("Step: %6d,\tLoss: %8.4f,\tAccuracy: %0.4f" % (count, loss_value, accuracy_value))
                count += 1
            except tf.errors.OutOfRangeError:
                break


        sess.run(dataset_iterator.initializer, feed_dict = {image: image_in, label: label_in})
        accuracy_value = sess.run(accuracy, feed_dict = {phase: False})
        print("Accuracy: ", accuracy_value)
