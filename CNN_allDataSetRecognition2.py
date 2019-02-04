import tensorflow as tf
import os
import random
import tensorflow.contrib.slim as slim
import time
import logging
import numpy as np
import pickle
from PIL import Image


logger = logging.getLogger('Training a chinese write char recognition')   #创建一个日志器
logger.setLevel(logging.INFO)                                                   #设置日志器将会处理的日志消息的最低严重级别
# formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch = logging.StreamHandler()                                                    # 将日志消息发送到输出到Stream，如std.out, std.err或任何file-like对象
ch.setLevel(logging.INFO)                                                       # 指定被处理的信息级别
logger.addHandler(ch)

#用户自定义的命令行参数,可在终端用 python CNN_allDataSetRecognition2.py -h 查看
tf.app.flags.DEFINE_boolean('random_flip_up_down', False, "Whether to random flip up down")
tf.app.flags.DEFINE_boolean('random_brightness', True, "whether to adjust brightness")
tf.app.flags.DEFINE_boolean('random_contrast', True, "whether to random constrast")

tf.app.flags.DEFINE_integer('charset_size', 3755, "Choose the first `charset_size` character to conduct our experiment.")
tf.app.flags.DEFINE_integer('image_size', 64, "Needs to provide same value as in training.")
tf.app.flags.DEFINE_boolean('gray', True, "whether to change the rbg to gray")
tf.app.flags.DEFINE_integer('max_steps', 12002, 'the max training steps ')   #in training, max_steps=200000
tf.app.flags.DEFINE_integer('eval_steps', 50, "the step num to eval")     #in training, eval_steps=1000
tf.app.flags.DEFINE_integer('save_steps', 2000, "the steps to save")     ##in training, save_steps=10000

tf.app.flags.DEFINE_string('checkpoint_dir', 'checkpoint/', 'the checkpoint dir')
tf.app.flags.DEFINE_string('train_data_dir', 'data/train/', 'the train dataset dir')
tf.app.flags.DEFINE_string('test_data_dir', 'data/test/', 'the test dataset dir')
tf.app.flags.DEFINE_string('log_dir', './log', 'the logging dir')

tf.app.flags.DEFINE_boolean('restore', False, 'whether to restore from checkpoint')
tf.app.flags.DEFINE_integer('epoch', 1, 'Number of epoches')
tf.app.flags.DEFINE_integer('batch_size', 128, 'Validation batch size')
tf.app.flags.DEFINE_string('mode', 'train', 'Running mode. One of {"train", "valid", "test"}')
FLAGS = tf.app.flags.FLAGS

##########--- 从filenames中异步读取文件，然后做shuffle ---########
class DataIterator:
    def __init__(self, data_dir):    #data_dir = 'data/train/'
        # Set FLAGS.charset_size to a small value if available computation power is limited.
        truncate_path = data_dir + ('%05d' % FLAGS.charset_size)    #  五个占位的格式 truncate_path：截断路径
        print(truncate_path)                                         # 'data/train/03755'
        self.image_names = []
                                                                              #iter=0                  iter=1            iter=2
                                                                #root        data/train/               data/train/00000   .../00001
                                                                #sub_folder  data/train/00000-03754    none              none
                                                                #files       none                      1958,3821,...     3054,4918
        for root, sub_folder, file_list in os.walk(data_dir):  #os.walk() 方法用于通过在目录树中游走输出在目录中的文件名，向上或者向下
            if root < truncate_path:
                self.image_names += [os.path.join(root, file_path) for file_path in file_list]     #把目录和文件名合成一个路径image_names = 'data/train/00000/1958'
        random.shuffle(self.image_names)                                                            #随机打乱次序
        self.labels = [int(file_name[len(data_dir):].split(os.sep)[0]) for file_name in self.image_names]   #根据路径取每一张图的标签 比如 data/train/00000/15->00000/15->00000
                                                                                                             #os.sep根据你所处的平台, 自动采用相应的分隔符号 / ou \
    @property
    def size(self):
        return len(self.labels)   #一共有多少类字

    @staticmethod     ## 静态方法无需实例化便可调用
    def data_augmentation(images):
        if FLAGS.random_flip_up_down:
            images = tf.image.random_flip_up_down(images)
        if FLAGS.random_brightness:
            images = tf.image.random_brightness(images, max_delta=0.3)
        if FLAGS.random_contrast:
            images = tf.image.random_contrast(images, 0.8, 1.2)
        return images

    def input_pipeline(self, batch_size, num_epochs=None, aug=False):
        images_tensor = tf.convert_to_tensor(self.image_names, dtype=tf.string)
        labels_tensor = tf.convert_to_tensor(self.labels, dtype=tf.int64)
        input_queue = tf.train.slice_input_producer([images_tensor, labels_tensor], num_epochs=num_epochs)  #num_epochs=None: 生成器可以无限次遍历tensor列表; 从 本地文件里抽取tensor，准备放入Filename Queue（文件名队列）中

        labels = input_queue[1]
        images_content = tf.read_file(input_queue[0])
        images = tf.image.convert_image_dtype(tf.image.decode_png(images_content, channels=1), tf.float32)  #channels 表示解码图像所需的颜色通道数量，=1：输出一个灰度图像
        #这里tf.image.decode_png 得到的是uint8格式，范围在0-255之间，经过convert_image_dtype 就会被转换为区间在0-1之间的float32格式
        if aug:
            with tf.name_scope('data_augmentation'):
                images = self.data_augmentation(images)
        new_size = tf.constant([FLAGS.image_size, FLAGS.image_size], dtype=tf.int32)   #[64,64]
        images = tf.image.resize_images(images, new_size)
        image_batch, label_batch = tf.train.shuffle_batch([images, labels], batch_size=batch_size, capacity=50000,
                                                          min_after_dequeue=10000)  #capacity给出队列的最大容量，当队列长度等于capacity时，TensorFlow将暂停入队操作，而只是等待元素出队。当元素个数小于容量时，TensorFlow将自动重新启动入队操作。
        return image_batch, label_batch                                            #batch_size表示每次出队的张量列表大小


def build_graph(top_k):
    # with tf.device('/cpu:0'):
    keep_prob = tf.placeholder(dtype=tf.float32, shape=[], name='keep_prob')
    images = tf.placeholder(dtype=tf.float32, shape=[None, 64, 64, 1], name='image_batch')
    labels = tf.placeholder(dtype=tf.int64, shape=[None], name='label_batch')
    with tf.name_scope('Conv1'):
        with tf.name_scope('conv2d_1'):
            conv_1 = slim.conv2d(images, 64, [3, 3], 1, padding='SAME', scope='conv1')  #1是步长  slim.conv2d中激活函数默认为ReLU
        with tf.name_scope('pool1'):
            max_pool_1 = slim.max_pool2d(conv_1, [2, 2], [2, 2], padding='SAME')  #后面的[2,2]是指步长
    with tf.name_scope('Conv2'):
        with tf.name_scope('conv2d_2'):
            conv_2 = slim.conv2d(max_pool_1, 128, [3, 3], padding='SAME', scope='conv2')
        with tf.name_scope('pool2'):
            max_pool_2 = slim.max_pool2d(conv_2, [2, 2], [2, 2], padding='SAME')
    with tf.name_scope('Conv3'):
        with tf.name_scope('conv2d_3'):
            conv_3 = slim.conv2d(max_pool_2, 256, [3, 3], padding='SAME', scope='conv3')
        with tf.name_scope('pool3'):
            max_pool_3 = slim.max_pool2d(conv_3, [2, 2], [2, 2], padding='SAME')

    with tf.name_scope('fc'):
        with tf.name_scope('flatten'):
            flatten = slim.flatten(max_pool_3)#卷积层和全连接层进行连接时有个输入格式转换过程，即卷积层的输出是一个x*y*z的矩阵，而全连接层的输入是一个向量，需要把矩阵拉直成向量
        with tf.name_scope('fc1'):
            fc1 = slim.fully_connected(slim.dropout(flatten, keep_prob), 1024, activation_fn=tf.nn.tanh, scope='fc1')
        with tf.name_scope('fc2'):
            logits = slim.fully_connected(slim.dropout(fc1, keep_prob), FLAGS.charset_size, activation_fn=None, scope='fc2')
        # logits = slim.fully_connected(flatten, FLAGS.charset_size, activation_fn=None, reuse=reuse, scope='fc')
    with tf.name_scope('loss'):
        loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels))  #传入的label为一个一维的vector，长度等于batch_size，每一个值的取值区间必须是[0，num_classes)，sparse_..函数首先将其转化为one-hot格式
        tf.summary.scalar('loss', loss)
    with tf.name_scope('accuracy'):
        accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(logits, 1), labels), tf.float32))
        tf.summary.scalar('accuracy', accuracy)

    global_step = tf.get_variable("step", [], initializer=tf.constant_initializer(0.0), trainable=False)  #创建一个新的变量  []shape,表明是一个标量
    rate = tf.train.exponential_decay(2e-4, global_step, decay_steps=2000, decay_rate=0.97, staircase=True) #设置学习率衰减，staircase=True表示每decay_steps轮训练后要乘以decay_rate
    train_op = tf.train.AdamOptimizer(learning_rate=rate).minimize(loss, global_step=global_step)  #以为全局步骤(global step)计数
    probabilities = tf.nn.softmax(logits)

    merged_summary_op = tf.summary.merge_all()
    predicted_val_top_k, predicted_index_top_k = tf.nn.top_k(probabilities, k=top_k)  #返回 probabilities 中每行最大的 k 个数，并且返回它们所在位置的索引
    with tf.name_scope('accuracy_in_top_k'):
        accuracy_in_top_k = tf.reduce_mean(tf.cast(tf.nn.in_top_k(probabilities, labels, top_k), tf.float32)) #probabilities中最大的K个值所对应的index是不是包括了labels: if top_k = 2, 3-0,4 4-0,5 labels=3 donc true
        tf.summary.scalar('accuracy_in_top_k', accuracy_in_top_k)

    return {'images': images,
            'labels': labels,
            'keep_prob': keep_prob,
            'top_k': top_k,
            'global_step': global_step,
            'train_op': train_op,
            'loss': loss,
            'accuracy': accuracy,
            'accuracy_top_k': accuracy_in_top_k,
            'merged_summary_op': merged_summary_op,
            'predicted_distribution': probabilities,
            'predicted_index_top_k': predicted_index_top_k,
            'predicted_val_top_k': predicted_val_top_k}


def train():
    print('Begin training')
    train_feeder = DataIterator(data_dir='data/train/')   #类DataIterator的实例化，调用初始化函数：__init__
    test_feeder = DataIterator(data_dir='data/test/')
    with tf.Session() as sess:
        train_images, train_labels = train_feeder.input_pipeline(batch_size=FLAGS.batch_size, aug=True)
        test_images, test_labels = test_feeder.input_pipeline(batch_size=FLAGS.batch_size)
        graph = build_graph(top_k=1)
        sess.run(tf.global_variables_initializer())
        coord = tf.train.Coordinator()  #创建一个线程管理器（协调器）对象，用来管理在Session中的多个线程
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)  #启动tensor的入队线程，用来启动多个工作线程同时将多个tensor（训练数据）推送入文件名称队列中
        saver = tf.train.Saver() #训练网络后想保存训练好的模型，以及在程序中读取以保存的训练好的模型,都要先实例化一个saver

        train_writer = tf.summary.FileWriter(FLAGS.log_dir + '/train', sess.graph)
        test_writer = tf.summary.FileWriter(FLAGS.log_dir + '/val')
        start_step = 0
        if FLAGS.restore:
            ckpt = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)  #自动获取最后一次保存的模型，重载模型的参数，继续训练或用于测试数据
            if ckpt:
                saver.restore(sess, ckpt)
                print("restore from the checkpoint {0}".format(ckpt))
                start_step += int(ckpt.split('-')[-1])

        logger.info(':::Training Start:::')
        try:
            while not coord.should_stop():#查询是否应该终止所有线程，当文件队列（queue）中的所有文件都已经读取出列的时候就应该停止Sesson中的所有线程了
                start_time = time.time()
                train_images_batch, train_labels_batch = sess.run([train_images, train_labels])   #线程管理：sess.run 来启动数据出列和执行计算
                feed_dict = {graph['images']: train_images_batch,    #feed_dict中变量对应placeholder
                             graph['labels']: train_labels_batch,
                             graph['keep_prob']: 0.8}
                _, loss_val, train_summary, step = sess.run(
                    [graph['train_op'], graph['loss'], graph['merged_summary_op'], graph['global_step']],
                    feed_dict=feed_dict)
                train_writer.add_summary(train_summary, step)
                end_time = time.time()
                logger.info("the step {0} takes {1} loss {2}".format(step, end_time - start_time, loss_val))
                if step > FLAGS.max_steps:    # max_steps=200000
                    break
                if step % FLAGS.eval_steps == 1:   #eval_steps=1000
                    test_images_batch, test_labels_batch = sess.run([test_images, test_labels])
                    feed_dict = {graph['images']: test_images_batch,
                                 graph['labels']: test_labels_batch,
                                 graph['keep_prob']: 1.0}
                    accuracy_test, test_summary = sess.run(
                        [graph['accuracy'], graph['merged_summary_op']],
                        feed_dict=feed_dict)
                    test_writer.add_summary(test_summary, step)
                    logger.info('===============Eval a batch=======================')
                    logger.info('the step {0} test accuracy: {1}'
                                .format(step, accuracy_test))      #每训练1000次，用一个batch的测试集进行测试并输出
                    logger.info('===============Eval a batch=======================')
                if step % FLAGS.save_steps == 1:   #save_steps=10000
                    logger.info('Save the ckpt of {0}'.format(step))
                    saver.save(sess, os.path.join(FLAGS.checkpoint_dir, 'my-model'),
                               global_step=graph['global_step'])#训练循环中，定期调用 saver.save() 方法，向文件夹中写入包含当前模型中所有可训练变量的 checkpoint 文件
        except tf.errors.OutOfRangeError:  # 线程管理中：一旦文件名队列空了之后，如果后面的流水线还要尝试从文件名队列中取出一个文件名，这将会触发OutOfRange错误
            logger.info('==================Train Finished================')
            saver.save(sess, os.path.join(FLAGS.checkpoint_dir, 'my-model'), global_step=graph['global_step'])
        finally:
            coord.request_stop()  #发出终止所有线程的命令
        coord.join(threads)  #把线程加入主线程，等待threads结束


def validation():
    print('validation')
    test_feeder = DataIterator(data_dir='data/test/')

    final_predict_val = []
    final_predict_index = []
    groundtruth = []

    with tf.Session() as sess:
        test_images, test_labels = test_feeder.input_pipeline(batch_size=FLAGS.batch_size, num_epochs=1)  #num_epochs=1：只用了一次完整的测试集进行测试
        graph = build_graph(3)    #top_k = 3

        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())  # initialize test_feeder's inside state

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        saver = tf.train.Saver()
        ckpt = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
        if ckpt:
            saver.restore(sess, ckpt)
            print("restore from the checkpoint {0}".format(ckpt))

        logger.info(':::Start validation:::')
        try:
            i = 0
            acc_top_1, acc_top_k = 0.0, 0.0
            while not coord.should_stop():
                i += 1
                start_time = time.time()
                test_images_batch, test_labels_batch = sess.run([test_images, test_labels])
                feed_dict = {graph['images']: test_images_batch,
                             graph['labels']: test_labels_batch,
                             graph['keep_prob']: 1.0}
                batch_labels, probs, indices, acc_1, acc_k = sess.run([graph['labels'],
                                                                       graph['predicted_val_top_k'],
                                                                       graph['predicted_index_top_k'],
                                                                       graph['accuracy'],
                                                                       graph['accuracy_top_k']], feed_dict=feed_dict)
                final_predict_val += probs.tolist()  #将数组或者矩阵转换成列表
                final_predict_index += indices.tolist()
                groundtruth += batch_labels.tolist()
                acc_top_1 += acc_1
                acc_top_k += acc_k
                end_time = time.time()
                logger.info("the batch {0} takes {1} seconds, accuracy = {2}(top_1) {3}(top_k)"
                            .format(i, end_time - start_time, acc_1, acc_k))

        except tf.errors.OutOfRangeError:
            logger.info('==================Validation Finished================')
            acc_top_1 = acc_top_1 * FLAGS.batch_size / test_feeder.size   #整个测试集平均下来的准确率
            acc_top_k = acc_top_k * FLAGS.batch_size / test_feeder.size
            logger.info('top 1 accuracy {0} top k accuracy {1}'.format(acc_top_1, acc_top_k))
        finally:
            coord.request_stop()
        coord.join(threads)
    return {'prob': final_predict_val, 'indices': final_predict_index, 'groundtruth': groundtruth}


def inference(image):
    print('inference')
    temp_image = Image.open(image).convert('L')  #转为灰度图
    temp_image = temp_image.resize((FLAGS.image_size, FLAGS.image_size), Image.ANTIALIAS) #Image.ANTIALIAS：高质量
    temp_image = np.asarray(temp_image) / 255.0
    temp_image = temp_image.reshape([-1, 64, 64, 1])
    with tf.Session() as sess:
        logger.info('========start inference============')
        # images = tf.placeholder(dtype=tf.float32, shape=[None, 64, 64, 1])
        # Pass a shadow label 0. This label will not affect the computation graph.
        graph = build_graph(top_k=3)
        saver = tf.train.Saver()
        ckpt = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
        if ckpt:
            saver.restore(sess, ckpt)
        predict_val, predict_index = sess.run([graph['predicted_val_top_k'], graph['predicted_index_top_k']],
                                              feed_dict={graph['images']: temp_image, graph['keep_prob']: 1.0})
    return predict_val, predict_index


def main(_):
    print(FLAGS.mode)
    if FLAGS.mode == "train":
        train()
    elif FLAGS.mode == 'validation':
        dct = validation()
        result_file = 'result.dict'
        logger.info('Write result into {0}'.format(result_file))
        with open(result_file, 'wb') as f:  #wb：以二进制写模式打开
            pickle.dump(dct, f)  #将结果数据流写入到文件对象中，pickle可以将对象以文件的形式存放在磁盘上
        logger.info('Write file ends')
    elif FLAGS.mode == 'inference':
        #image_path = 'data/test/00190/13320.png'
        image_path = 'data/test/00190/12621.png'
        final_predict_val, final_predict_index = inference(image_path)
        logger.info('the result info label {0} predict index {1} predict_val {2}'.format(190, final_predict_index,
                                                                                         final_predict_val))

if __name__ == "__main__":
    tf.app.run()
