import tensorflow as tf
import numpy as np
from scipy.misc import imread, imresize, imsave
from PIL import Image
import os
import random

# START_VOCA = [UNK]
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 304875
NUM_EXAMPLES_PER_EPOCH_FOR_TEST = 33876
VOCUBLARY_SIZE = 5805
IMAGE_WHITE = 256
IMAGE_HEIGHT = 32
IMAGE_CHANNElS = 3


# 读取数据并保存在一个字典里
def read_text(filename):
    vocublary = {}
    index = 0
    f = open(filename)
    line = f.readline()
    for s in line:
        vocublary[s] = index
        index = index + 1
    f.close()
    print('size of the vocublary: ', len(vocublary))
    return vocublary


def resize_img(img, fixed_height):
    '''
    resize the image
    @param img: the input image
    @param fixed_height: a fixed_height of the image, here is 32
    @return: the new width and height of resized image
    '''
    width, height = img.size()
    if height > fixed_height:
        ratio = float(fixed_height) / height
        width = width * ratio
        height = fixed_height
    else:
        ratio = float(fixed_height) / height
        width = width * ratio
        height = fixed_height
    return width, height


# 生成TFRecord
def generation_TFRecord(data_base_dir, vocub_file_name):
    vocublary = read_text(vocub_file_name)

    image_name_list = []
    for file in os.listdir(data_base_dir):
        if file.endswith('.jpg'):
            image_name_list.append(file)

    random.shuffle(image_name_list)
    image_capacity = len(image_name_list)
    print(len(image_name_list))

    # 生成train tfrecord
    train_writer = tf.python_io.TFRecordWriter('./dataset/train_dataset.tfrecords')
    train_imgae_name_list = image_name_list[0:int(image_capacity * 0.9)]
    print(len(train_imgae_name_list))
    for train_image_name in train_imgae_name_list:
        train_image_label = []
        for s in train_image_name.strip('.jpg'):
            train_image_label.append(vocublary[s])
        # print(train_image_label)

        img = Image.open(os.path.join(data_base_dir, train_image_name))
        height, width = resize_img(img, 32)
        # img.show()
        # 将图片转换为二进制形式
        img_raw = img.tobytes()
        # Example对象对label和image进行封装
        example = tf.train.Example(features=tf.train.Features(feature={
            'label': tf.train.Feature(int64_list=tf.train.Int64List(value=train_image_label)),
            'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw])),
            # 'width': tf.train.Feature(int64_list=tf.train.Int64List(value=width)),
            # 'height': tf.train.Feature(int64_list=tf.train.Int64List(value=height))
        }))
        # 序列转换成字符串
        train_writer.write(example.SerializeToString())
    train_writer.close()

    # 生成test tfrecord
    test_writer = tf.python_io.TFRecordWriter('./dataset/test_dataset.tfrecords')
    test_image_name_list = image_name_list[int(image_capacity * 0.9):image_capacity]
    print(len(test_image_name_list))
    for test_image_name in test_image_name_list:
        test_image_label = []
        for s in test_image_name.strip('.jpg'):
            test_image_label.append(vocublary[s])

        img = Image.open(os.path.join(data_base_dir, test_image_name))
        height, width = resize_img(img, 32)
        img_raw = img.tobytes()
        example = tf.train.Example(features=tf.train.Features(feature={
            'label': tf.train.Feature(int64_list=tf.train.Int64List(value=test_image_label)),
            'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw])),
            # 'width': tf.train.Feature(int64_list=tf.train.Int64List(value=width)),
            # 'height': tf.train.Feature(int64_list=tf.train.Int64List(value=height))
        }))
        test_writer.write(example.SerializeToString())
    test_writer.close()


# 读取tfrecord文件
def read_and_decode(filename):
    # 生成一个queue队列
    filename_queue = tf.train.string_input_producer([filename])

    reader = tf.TFRecordReader()
    # 返回文件名和文件
    _, serialized_example = reader.read(filename_queue)
    # 将image数据和label取出来
    features = tf.parse_single_example(serialized=serialized_example,
                                       features={'label': tf.FixedLenFeature([], tf.int64),
                                                 'img_raw': tf.FixedLenFeature([], tf.string)})

    img = tf.decode_raw(features['img_raw'], tf.uint8)
    img = tf.reshape(img, shape=[IMAGE_WHITE, IMAGE_HEIGHT, 3])
    # 在流中抛出img张量
    img = tf.cast(img, tf.float32) * (1. / 255) - 0.5

    # 在流中抛出label张量
    label = tf.cast(features['label'], tf.int32)
    label = tf.one_hot(label, VOCUBLARY_SIZE, on_value=1.0, off_value=0.0)
    # print('image:', img)
    # print('label', label)
    return img, label


def CNN_VGG(inputs):
    ''' CNN extract feature from each input image
    @param inputs: the input image
    @return: feature maps
    '''
    with tf.variable_scope('VGG_CNN'):
        conv1 = tf.layers.conv2d(inputs=inputs, filters=64, kernel_size=(3, 3),
                                 padding='SAME', activation=tf.nn.relu, name='conv_1')
        pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=(2, 2), strides=2, name='pool_1')

        conv2 = tf.layers.conv2d(inputs=pool1, filters=128, kernel_size=(3, 3),
                                 padding='SAME', activation=tf.nn.relu, name='conv_2')
        pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=(2, 2), strides=2, name='pool_2')

        conv3 = tf.layers.conv2d(inputs=pool2, filters=256, kernel_size=(3, 3),
                                 padding='SAME', activation=tf.nn.relu, name='conv_3')

        conv4 = tf.layers.conv2d(inputs=conv3, filters=256, kernel_size=(3, 3),
                                 padding='SAME', activation=tf.nn.relu, name='conv_4')
        pool3 = tf.layers.max_pooling2d(inputs=conv4, pool_size=(1, 2), strides=2, name='pool_3')

        conv5 = tf.layers.conv2d(inputs=pool3, filters=512, kernel_size=(3, 3),
                                 padding='SAME', activation=tf.nn.relu, name='conv_5')
        bn1 = tf.layers.batch_normalization(conv5, name='bn1')

        conv6 = tf.layers.conv2d(inputs=bn1, filters=512, kernel_size=(3, 3),
                                 padding='SAME', activation=tf.nn.relu, name='conv_6')
        bn2 = tf.layers.batch_normalization(conv6, name='bn_2')
        pool4 = tf.layers.max_pooling2d(inputs=bn2, pool_size=(1, 2), strides=2, name='pool_4')

        conv7 = tf.layers.conv2d(inputs=pool4, filters=512, kernel_size=(3, 3),
                                 padding='SAME', activation=tf.nn.relu, name='conv_7')
    return conv7


def main(argv):
    # generation_TFRecord('./train_val_dataset', './out.txt')
    img, label = read_and_decode('./dataset/test_dataset.tfrecords')
    print(img.shape, label.shape)

    batch_size = 32
    min_fraction_of_examples_in_queue = 0.4
    min_queue_examples = int(min_fraction_of_examples_in_queue * NUM_EXAMPLES_PER_EPOCH_FOR_TEST)
    test_img_batch, test_label_batch = tf.train.shuffle_batch([img, label],
                                                              batch_size=batch_size,
                                                              capacity=min_queue_examples + 3 * batch_size,
                                                              min_after_dequeue=min_queue_examples,
                                                              num_threads=32)
    print(test_img_batch.shape, test_label_batch.shape)

    logits = CNN_VGG(test_img_batch)
    print(logits.shape)
    logits = tf.layers.flatten(logits)
    print(logits.shape)
    weight = tf.Variable(tf.random_uniform([16384, 5805], 0.0, 1.0), dtype=tf.float32)
    logits = tf.matmul(logits, weight)
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=test_label_batch)
    cross_entropy_loss = tf.reduce_mean(cross_entropy)
    optimizer = tf.train.AdadeltaOptimizer(0.001).minimize(cross_entropy_loss)

    with tf.Session() as session:
        session.run(tf.global_variables_initializer())
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=session, coord=coord)
        for index in range(10):
            _, loss = session.run([optimizer, cross_entropy_loss])
            print('loss:', loss)
            # batch_image, batch_label = session.run([test_img_batch, test_label_batch])
            # print(batch_image.shape)
            # print(batch_label.shape)

        coord.request_stop()
        coord.join(threads=threads)


if __name__ == '__main__':
    tf.app.run()
