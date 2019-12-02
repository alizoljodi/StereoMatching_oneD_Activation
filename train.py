import tensorflow as tf
import os
import models.net_factory as nf
import numpy as np
from data_handler import Data_handler
from simanneal import Annealer
import math
import sqlite3
import random
flags = tf.app.flags

flags.DEFINE_integer('batch_size', 128, 'Batch size.')
flags.DEFINE_integer('num_iter', 40000, 'Total training iterations')
flags.DEFINE_string('model_dir', 'model', 'Trained network dir')
flags.DEFINE_string('data_version', 'kitti2012', 'kitti2012 or kitti2015')
flags.DEFINE_string('data_root', '', 'training dataset dir')
flags.DEFINE_string('util_root', '', 'Binary training files dir')
flags.DEFINE_string('net_type', 'win37_dep9', 'Network type: win37_dep9 pr win19_dep9')

flags.DEFINE_integer('eval_size', 200, 'number of evaluation patchs per iteration')
flags.DEFINE_integer('num_tr_img', 160, 'number of training images')
flags.DEFINE_integer('num_val_img', 34, 'number of evaluation images')
flags.DEFINE_integer('patch_size', 37, 'training patch size')
flags.DEFINE_integer('num_val_loc', 50000, 'number of validation locations')
flags.DEFINE_integer('disp_range', 201, 'disparity range')
flags.DEFINE_string('phase', 'train', 'train or evaluate')

FLAGS = flags.FLAGS

np.random.seed(123)

dhandler = Data_handler(data_version=FLAGS.data_version,
                        data_root=FLAGS.data_root,
                        util_root=FLAGS.util_root,
                        num_tr_img=FLAGS.num_tr_img,
                        num_val_img=FLAGS.num_val_img,
                        num_val_loc=FLAGS.num_val_loc,
                        batch_size=FLAGS.batch_size,
                        patch_size=FLAGS.patch_size,
                        disp_range=FLAGS.disp_range)

if FLAGS.data_version == 'kitti2012':
    num_channels = 1
elif FLAGS.data_version == 'kitti2015':
    num_channels = 3
else:
    sys.exit('data_version should be either kitti2012 or kitti2015')

class ActivationAnnealer(Annealer):
    def __init__(self,state):
        self.num = 0
        self.best = math.inf
        self.path = FLAGS.model_dir + '\\bests.db'
        conn = sqlite3.connect(self.path)
        c = conn.cursor()
        c.execute('''CREATE TABLE bestss
                             (num int, Act_list text, acc real, energy real)''')
        conn.commit()
        c = conn.cursor()
        c.execute('''CREATE TABLE _all_
                                    (num int, Act_list text, acc real,energy real)''')
        conn.commit()
        conn.close()
        super(ActivationAnnealer, self).__init__(state)
    def move(self):
        index=random.randint(len(self.state))
        Activations=['sigmoid','relu','elu']
        self.state[index]=random.choice(Activations)
        return self.energy()
    def energy(self):
        train(self.state,self.num)
        acc=evaluate(self.state,self.num)
        conn = sqlite3.connect(self.path)
        c = conn.cursor()
        statea=str(self.state)
        if acc==0:
            e=math.inf
        else:
            e=1/acc
        c.execute('''INSERT INTO _all_ VALUES (?,?,?,?)''', [self.num, statea, acc, e])
        conn.commit()
        conn.close()

        if e < self.best:
            conn = sqlite3.connect(self.path)
            c = conn.cursor()
            c.execute('''INSERT INTO bestss VALUES (?,?,?,?)''', [self.num, statea, acc, e])
            conn.commit()
            conn.close()
            self.best = e
            #print(colored('4ogmmregreomgerogmreomerormfrofmfoemwfoewmfewofm', 'red'))
        self.num = self.num + 1
        #print(str(now))
        print(self.state)

        return e


def train(state,num):
    path=FLAGS.model_dir+'\\num'
    if not os.path.exists(path):
        os.makedirs(path)
    tf.reset_default_graph()
    run_meta = tf.RunMetadata()

    g = tf.Graph()
    strategy = tf.distribute.MirroredStrategy()
    with strategy.scope():
        with g.as_default():

            limage = tf.placeholder(tf.float32, [None, FLAGS.patch_size, FLAGS.patch_size, num_channels], name='limage')
            rimage = tf.placeholder(tf.float32,
                                    [None, FLAGS.patch_size, FLAGS.patch_size + FLAGS.disp_range - 1, num_channels],
                                    name='rimage')
            targets = tf.placeholder(tf.float32, [None, FLAGS.disp_range], name='targets')

            snet = nf.create(limage, rimage, targets,state, FLAGS.net_type)

            loss = snet['loss']
            train_step = snet['train_step']
            session = tf.InteractiveSession()
            session.run(tf.global_variables_initializer())
            saver = tf.train.Saver(max_to_keep=1)

            acc_loss = tf.placeholder(tf.float32, shape=())
            loss_summary = tf.summary.scalar('loss', acc_loss)
            train_writer = tf.summary.FileWriter(FLAGS.model_dir + '/training', g)

            saver = tf.train.Saver(max_to_keep=1)
            losses = []
            summary_index = 1
            lrate = 1e-2

            for it in range(1, FLAGS.num_iter):
                lpatch, rpatch, patch_targets = dhandler.next_batch()

                train_dict = {limage: lpatch, rimage: rpatch, targets: patch_targets,
                              snet['is_training']: True, snet['lrate']: lrate}
                _, mini_loss = session.run([train_step, loss], feed_dict=train_dict)
                losses.append(mini_loss)

                if it % 100 == 0:
                    print('Loss at step: %d: %.6f' % (it, mini_loss))
                    saver.save(session, os.path.join(FLAGS.model_dir, 'model.ckpt'), global_step=snet['global_step'])
                    train_summary = session.run(loss_summary,
                                                feed_dict={acc_loss: np.mean(losses)})
                    train_writer.add_summary(train_summary, summary_index)
                    summary_index += 1
                    train_writer.flush()
                    losses = []

                if it == 24000:
                    lrate = lrate / 5.
                elif it > 24000 and (it - 24000) % 8000 == 0:
                    lrate = lrate / 5.


def evaluate(state,num):
    lpatch, rpatch, patch_targets = dhandler.evaluate()
    labels = np.argmax(patch_targets, axis=1)
    path=FLAGS.model_dir+'\\num'

    with tf.Session() as session:
        limage = tf.placeholder(tf.float32, [None, FLAGS.patch_size, FLAGS.patch_size, num_channels], name='limage')
        rimage = tf.placeholder(tf.float32,
                                [None, FLAGS.patch_size, FLAGS.patch_size + FLAGS.disp_range - 1, num_channels],
                                name='rimage')
        targets = tf.placeholder(tf.float32, [None, FLAGS.disp_range], name='targets')

        snet = nf.create(state,limage, rimage, targets, FLAGS.net_type)
        prod = snet['inner_product']
        predicted = tf.argmax(prod, axis=1)
        acc_count = 0

        saver = tf.train.Saver()
        saver.restore(session, tf.train.latest_checkpoint(path))

        for i in range(0, lpatch.shape[0], FLAGS.eval_size):
            eval_dict = {limage: lpatch[i: i + FLAGS.eval_size],
                         rimage: rpatch[i: i + FLAGS.eval_size], snet['is_training']: False}
            pred = session.run([predicted], feed_dict=eval_dict)
            acc_count += np.sum(np.abs(pred - labels[i: i + FLAGS.eval_size]) <= 3)
            print('iter. %d finished, with %d correct (3-pixel error)' % (i + 1, acc_count))

        print('accuracy: %.3f' % ((acc_count / lpatch.shape[0]) * 100))
        return ((acc_count / lpatch.shape[0]) * 100)


if __name__=='__main__':
    init=['relu','relu','relu','relu','relu','relu','relu','relu','relu','relu','relu','relu','relu']

    tsp=ActivationAnnealer(init)
    tsp.set_schedule(tsp.auto(0.01, 10))
    tsp.copy_strategy = "slice"
    state, e = tsp.anneal()
    print()
    print("%i mile rout:" % e)
    print("state {}".format(state))




