import tensorflow as tf

slim = tf.contrib.slim


def create_network(inputs, is_training, scope="win37_dep9", reuse=False):
    num_maps = 64
    kw = 5
    kh = 5

    with tf.variable_scope(scope, reuse=reuse):
        with slim.arg_scope([slim.conv2d], padding='VALID', activation_fn=tf.nn.relu,
                            normalizer_fn=slim.batch_norm, normalizer_params={'is_training': is_training}):
            net=slim.conv2d(inputs,64,[5,5],padding='same',scope='conv_1')
            net=slim.conv2d(net,64,[3,3],padding='valid',scope='conv_2')
            net=slim.conv2d(net,64,[5,5],padding='same',scope='conv_3')
            net=slim.conv2d(net,32,[3,3],padding='valid',scope='conv_4')
            net = slim.conv2d(net, 32, [3, 3], padding='valid', scope='conv_5')
            net = slim.conv2d(net, 32, [5,5], padding='valid', scope='conv_6')
            net = slim.conv2d(net, 64, [5,5], padding='same', scope='conv_7')
            net=slim.batch_norm(net,is_training=is_training,scope='batch_1')
            net = slim.conv2d(net, 32, [5,5], padding='valid', scope='conv_8')
            net = slim.conv2d(net, 32, [5,5], padding='same', scope='conv_9')
            net = slim.conv2d(net, 32, [7,7], padding='valid', scope='conv_10')
            net = slim.batch_norm(net, is_training=is_training,scope='batch_2')
            net = slim.batch_norm(net, is_training=is_training,scope='batch_3')
            net = slim.conv2d(net, 32, [9,9], padding='valid', scope='conv_11')
            net = slim.conv2d(net, 32, [7, 7], padding='valid', scope='conv_12')
            net = slim.conv2d(net, 32, [3,3], padding='valid', scope='conv_13')
            net = slim.batch_norm(net, is_training=is_training, scope='batch_4')
            '''net = slim.conv2d(inputs, 32, [kh, kw], scope='conv_bn_relu1')
            net = slim.conv2d(net, 32, [kh, kw], scope='conv_bn_relu2')
            net = slim.repeat(net, 6, slim.conv2d, num_maps, [kh, kw], scope='conv_bn_relu3_8')

            net = slim.conv2d(net, num_maps, [kh, kw], scope='conv9', activation_fn=None,
                              normalizer_fn=None)
            net = slim.batch_norm(net, is_training=is_training)'''


    return net



