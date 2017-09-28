import os
import sys
import pickle
import numpy as np
import tensorflow as tf
import random
import data
import builder


class ShufflenetClassifier(object):
    def __init__(self,
                 n_classes,
                 input_resolution,
                 multiclass=True,
                 batch_norm=True):
        self.n_classes = n_classes
        self.input_resolution = input_resolution
        self.batch_norm = batch_norm
        self.multiclass = multiclass
        self.builder = builder.shufflenet.ShuffleNetBuilder()
        self.data_loader = data.pascal.PascalVocDataLoader()

        # dataset variables
        self.minibatch_loader = None
        self.test_samples = None

        # graph variables
        self.inputs = None
        self.true_outputs = None
        self.is_training = None
        self.input_keepprob = None
        self.conv_keepprob = None
        self.fc_keepprob = None
        self.optimiser = None
        self.loss = None
        self.accuracy = None
        self.summary_op = None
        self.graph = None

    def initialize_classifier(self):
        self.data_loader.initialize_data()
        sample_names = self.data_loader.load_sample_names()
        random.shuffle(sample_names)
        self.test_samples = sample_names[0:50]
        train_samples = sample_names[50:]
        self.minibatch_loader = data.base.MulticlassMinibatchLoader(self.data_loader, train_samples, 20)

    def _save_checkpoint(self, path):
        pass

    def _save_datasets(self, path):
        pass

    def save_classifier(self, path):
        pass

    def _load_checkpoint(self, path):
        pass

    def _load_datasets(self, path):
        pass

    def load_classifier(self, path):
        pass

    def build_graph(self):
        self.graph = tf.Graph()

        with self.graph.as_default():
            inputs = tf.placeholder(tf.float32, [None, *self.input_resolution, 3])
            true_outputs = tf.placeholder(tf.float32, [None, self.n_classes])
            if self.batch_norm:
                is_training = tf.placeholder(tf.bool)
            else:
                is_training = None
            input_keepprob = tf.placeholder(self.builder.dtype)
            conv_keepprob = tf.placeholder(self.builder.dtype)
            fc_keepprob = tf.placeholder(self.builder.dtype)

            segment_tails = self.builder.add_trunk(inputs,  # input
                                                   first_conv_ksize=3,  # kernel size of first convolution
                                                   first_conv_stride=2,  # stride of first convolution
                                                   first_conv_kernels=24,  # number of kernels in first convolution layer
                                                   first_pool_ksize=3,  # kernel size of first maxpool
                                                   first_pool_stride=2,  # stride of first maxpool
                                                   batch_norm=self.batch_norm,  # if True, batch normalization is added
                                                   is_training=is_training,  # pass variable indicating if network is training
                                                   shuffle_segments=[4, 8, 4],
                                                   n_groups=8,
                                                   base_channels=384,
                                                   bottleneck_ratio=.25,
                                                   input_keepprob=input_keepprob,  # input keep probability for dropout
                                                   conv_keepprob=conv_keepprob  # conv keep probability for dropout
                                                   )

            pred_outputs = self.builder.add_classifier_head(segment_tails[-1],  # output from the trunk
                                                            self.n_classes,  # number of outputs
                                                            fc_keepprob=fc_keepprob  # fc keep probability for dropout
                                                            )

            if self.multiclass:
                pred_outputs = tf.reshape(pred_outputs, [-1, self.n_classes])
            else:
                pred_outputs = tf.reshape(pred_outputs, [-1, self.n_classes])

            # training
            if self.multiclass:
                cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=true_outputs, logits=pred_outputs)
            else:
                cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=true_outputs, logits=pred_outputs)
            loss = tf.reduce_mean(cross_entropy)
            tf.summary.scalar('loss', loss)

            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with self.graph.control_dependencies(update_ops):
                optimiser = tf.train.AdamOptimizer(learning_rate=1e-3, epsilon=1e-8).minimize(loss)

            # testing
            correct_pred = tf.equal(tf.argmax(pred_outputs, 1), tf.argmax(true_outputs, 1))
            accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
            tf.summary.scalar('accuracy', accuracy)

            summary_op = tf.summary.merge_all()

            self.inputs = inputs
            self.true_outputs = true_outputs
            self.is_training = is_training
            self.input_keepprob = input_keepprob
            self.conv_keepprob = conv_keepprob
            self.fc_keepprob = fc_keepprob
            self.optimiser = optimiser
            self.loss = loss
            self.accuracy = accuracy
            self.summary_op = summary_op

    def train(self):
        # TODO: properly define these

        train_dir = ""
        valid_dir = ""
        latest_checkpoints_dir = ""
        best_checkpoints_dir = ""
        state_path = ""

        with self.graph.as_default():
            # Init the variables
            init = tf.global_variables_initializer()

            # Prepare to save checkpoints
            latest_checkpoint_saver = tf.train.Saver()
            best_checkpoint_saver = tf.train.Saver()

            # summaries

            # init dirs
            if not os.path.isdir(train_dir):
                os.makedirs(train_dir)

            if not os.path.isdir(valid_dir):
                os.makedirs(valid_dir)

            if not os.path.isdir(latest_checkpoints_dir):
                os.makedirs(latest_checkpoints_dir)

            if not os.path.isdir(best_checkpoints_dir):
                os.makedirs(best_checkpoints_dir)

            # summary writers
            train_writer = tf.summary.FileWriter(train_dir, graph=tf.get_default_graph())
            valid_writer = tf.summary.FileWriter(valid_dir, graph=tf.get_default_graph())

            max_epochs = 100000
            batch_size = 10
            report_epochs = 25

            best_valid_loss = sys.float_info.max

            with tf.Session() as sess:
                step = 1

                if os.path.exists(state_path):
                    state = pickle.load(open(state_path, "rb"))
                    step = state["step"]

                if os.path.exists(os.path.join(latest_checkpoints_dir, "checkpoint")):
                    latest_checkpoint_saver.restore(sess, tf.train.latest_checkpoint(latest_checkpoints_dir))
                else:
                    sess.run(init)

                while step < max_epochs:
                    batch_samples, batch_labels = self.minibatch_loader.random_minibatch()
                    train_loss, train_acc, _, train_summary = sess.run([self.loss,
                                                                        self.accuracy,
                                                                        self.optimiser,
                                                                        self.summary_op],
                                                                       feed_dict={self.inputs: batch_samples,
                                                                                  self.true_outputs: batch_labels,
                                                                                  self.is_training: True,
                                                                                  self.input_keepprob: .95,
                                                                                  self.conv_keepprob: .9,
                                                                                  self.fc_keepprob: .6})
                    train_writer.add_summary(train_summary, step)
                    train_writer.flush()

                    state = {
                        "step": step
                    }
                    pickle.dump(state, open(state_path, "wb"))

                    if step % report_epochs == 0:
                        valid_loss, valid_acc, valid_summary, valid_logits, valid_labels = sess.run(
                            [loss, accuracy, summary_op, logits, true_outputs],
                            feed_dict={inputs: valid_dataset,
                                       true_outputs: valid_labels,
                                       is_training: False,
                                       input_keepprob: 1,
                                       conv_keepprob: 1,
                                       fc_keepprob: 1})

                        print("Epoch " + str(step))
                        print("Training Loss={:.6f}".format(train_loss))
                        print("Training Accuracy={:.6f}".format(train_acc))
                        # pred_labels = np.argmax(train_logits, 1)
                        # true_labels = np.argmax(batch_labels, 1)
                        # for i in range(l en(pred_labels)):
                        #     print("%s==%s" % (pred_labels[i], true_labels[i]))
                        # print

                        print("Valid Loss={:.6f}".format(valid_loss))
                        print("Valid Accuracy={:.6f}".format(valid_acc))
                        # pred_labels = np.argmax(valid_logits, 1)
                        # true_labels = np.argmax(valid_labels, 1)
                        # for i in range(len(pred_labels)):
                        #     print("%s==%s" % (pred_labels[i], true_labels[i]))
                        # print

                        valid_writer.add_summary(valid_summary, step)
                        valid_writer.flush()

                        latest_checkpoint_saver.save(sess, latest_checkpoint_path)

                        if valid_loss > best_valid_loss:
                            best_valid_loss = valid_loss
                            best_checkpoint_saver.save(sess, best_checkpoint_path)

                    step += 1

                print("Training finished!")
                test_loss, test_acc = sess.run([loss, accuracy],
                                               feed_dict={inputs: test_dataset,
                                                          true_outputs: test_labels,
                                                          is_training: False,
                                                          input_keepprob: 1,
                                                          conv_keepprob: 1,
                                                          fc_keepprob: 1})

                print("Testation Loss=" + "{:.6f}".format(test_loss))
                print("Testation Accuracy=" + "{:.6f}".format(test_acc))
