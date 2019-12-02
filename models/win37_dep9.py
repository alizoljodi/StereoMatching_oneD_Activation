import tensorflow as tf

slim = tf.contrib.slim


def create_network(state,inputs, is_training, scope="win37_dep9", reuse=False):
    num_maps = 64
    kw = 5
    kh = 5

    with tf.variable_scope(scope, reuse=reuse):
        with slim.arg_scope([slim.conv2d], padding='VALID', activation_fn=tf.nn.relu,
                            normalizer_fn=slim.batch_norm, normalizer_params={'is_training': is_training}):
            act1=activation_maker(state.pop(0))
            net=slim.conv2d(inputs,64,[5,5],padding='same',activation_fn=act1,scope='conv_1')
            act2 = activation_maker(state.pop(0))
            net=slim.conv2d(net,64,[3,3],padding='valid',activation_fn=act2,scope='conv_2')
            act3 = activation_maker(state.pop(0))
            net=slim.conv2d(net,64,[5,5],padding='same',activation_fn=act3,scope='conv_3')
            act4 = activation_maker(state.pop(0))
            net=slim.conv2d(net,32,[3,3],padding='valid',activation_fn=act4,scope='conv_4')
            act5 = activation_maker(state.pop(0))
            net = slim.conv2d(net, 32, [3, 3], padding='valid',activation_fn=act5, scope='conv_5')
            act6 = activation_maker(state.pop(0))
            net = slim.conv2d(net, 32, [5,5], padding='valid',activation_fn=act6, scope='conv_6')
            act7 = activation_maker(state.pop(0))
            net = slim.conv2d(net, 64, [5,5], padding='same',activation_fn=act7, scope='conv_7')
            net=slim.batch_norm(net,is_training=is_training,scope='batch_1')
            act8 = activation_maker(state.pop(0))
            net = slim.conv2d(net, 32, [5,5], padding='valid',activation_fn=act8, scope='conv_8')
            act9 = activation_maker(state.pop(0))
            net = slim.conv2d(net, 32, [5,5], padding='same',activation_fn=act9, scope='conv_9')
            act10 = activation_maker(state.pop(0))
            net = slim.conv2d(net, 32, [7,7], padding='valid',activation_fn=act10, scope='conv_10')
            net = slim.batch_norm(net, is_training=is_training,scope='batch_2')
            net = slim.batch_norm(net, is_training=is_training,scope='batch_3')
            act11 = activation_maker(state.pop(0))
            net = slim.conv2d(net, 32, [9,9], padding='valid',activation_fn=act11, scope='conv_11')
            act12 = activation_maker(state.pop(0))
            net = slim.conv2d(net, 32, [7, 7], padding='valid',activation_fn=act12, scope='conv_12')
            act13 = activation_maker(state.pop(0))
            net = slim.conv2d(net, 32, [3,3], padding='valid',activation_fn=act13, scope='conv_13')
            act14 = activation_maker(state.pop(0))
            net = slim.batch_norm(net, is_training=is_training,activation_fn=act14, scope='batch_4')
            '''net = slim.conv2d(inputs, 32, [kh, kw], scope='conv_bn_relu1')
            net = slim.conv2d(net, 32, [kh, kw], scope='conv_bn_relu2')
            net = slim.repeat(net, 6, slim.conv2d, num_maps, [kh, kw], scope='conv_bn_relu3_8')

            net = slim.conv2d(net, num_maps, [kh, kw], scope='conv9', activation_fn=None,
                              normalizer_fn=None)
            net = slim.batch_norm(net, is_training=is_training)'''


    return net
def activation_maker(gen):
    if gen=='relu':
        return tf.nn.relu
    elif gen=='sigmoid':
        return tf.nn.sigmoid
    elif gen=='elu':
        return tf.nn.elu
    else:
        raise ValueError('non appropriate gen')



