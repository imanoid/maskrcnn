import os
import sys
import pickle
import numpy as np
import tensorflow as tf
import random
import data
import builder
import util


class ShufflenetClassifier(object):
    def __init__(self,
                 name,
                 n_classes,
                 input_resolution,
                 multiclass=True,
                 batch_norm=True):
        self.name = name
        self.n_classes = n_classes
        self.input_resolution = input_resolution
        self.batch_norm = batch_norm
        self.multiclass = multiclass
        self.builder = builder.shufflenet.ShuffleNetBuilder()
        self.data_loader = \
            data.pascal.PascalVocDataLoader(config_name="{}_dataloader".format(name),
                                            voc_dir="/media/imanoid/Data/workspace/data/VOCdevkit/VOC2012",
                                            image_shape=input_resolution)

        # dataset variables
        self.train_minibatch_loader = None
        self.test_minibatch_loader = None

        # graph variables
        self.inputs = None
        self.true_outputs = None
        self.pred_class = None
        self.is_training = None
        self.input_keepprob = None
        self.conv_keepprob = None
        self.fc_keepprob = None
        self.optimiser = None
        self.loss = None
        self.accuracy = None
        self.summary_op = None
        self.graph = None

        # paths
        self.train_dir = None
        self.valid_dir = None
        self.latest_checkpoints_dir = None
        self.best_checkpoints_dir = None
        self.train_state_path = None

    def _init_samples(self):
        # self.data_loader.initialize_data()
        sample_names = self.data_loader.load_sample_names("multiclass")
        random.shuffle(sample_names)
        self.test_samples = sample_names[0:20]
        self.train_samples = sample_names[20:]

    def _init_minibatch_loaders(self):
        self.train_minibatch_loader = data.minibatch.MinibatchLoader(self.data_loader, self.train_samples, 20)
        self.test_minibatch_loader = data.minibatch.MinibatchLoader(self.data_loader, self.test_samples, 20)

    def _init_paths(self):
        tensorlog_dir = "/media/imanoid/Data/workspace/data/tensorlog/"
        run_log = os.path.join(tensorlog_dir, self.name)
        self.train_dir = os.path.join(run_log, "train")
        self.valid_dir = os.path.join(run_log, "valid")
        self.latest_checkpoints_dir = os.path.join(run_log, "latest_checkpoints")
        self.best_checkpoints_dir = os.path.join(run_log, "best_checkpoints")
        self.train_state_path = os.path.join(run_log, "state.pickle")

        if not os.path.isdir(self.train_dir):
            os.makedirs(self.train_dir)

        if not os.path.isdir(self.valid_dir):
            os.makedirs(self.valid_dir)

        if not os.path.isdir(self.latest_checkpoints_dir):
            os.makedirs(self.latest_checkpoints_dir)

        if not os.path.isdir(self.best_checkpoints_dir):
            os.makedirs(self.best_checkpoints_dir)

    def _init_graph(self):
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
            self.pred_class = tf.argmax(pred_outputs, 1)
            true_class = tf.argmax(true_outputs, 1)
            correct_pred = tf.equal(self.pred_class, true_class)
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

            self.checkpoint_saver = tf.train.Saver()

    def _init_summary_writers(self):
        # summary writers
        self.train_writer = tf.summary.FileWriter(self.train_dir, graph=self.graph)
        self.valid_writer = tf.summary.FileWriter(self.valid_dir, graph=self.graph)

    def initialize_classifier(self):
        self._init_samples()
        self._init_minibatch_loaders()
        self._init_paths()
        self._init_graph()
        self._init_summary_writers()

    def _load_checkpoint(self, path, session):
        self.checkpoint_saver.restore(session, tf.train.latest_checkpoint(path))

    def _save_checkpoint(self, path, session):
        self.checkpoint_saver.save(session, path)

    def _latest_checkpoint_exists(self):
        return os.path.exists(os.path.join(self.latest_checkpoints_dir, "checkpoint"))

    def _load_latest_checkpoint(self, session):
        self._load_checkpoint(self.latest_checkpoints_dir, session)

    def _save_latest_checkpoint(self, session):
        self._save_checkpoint(self.latest_checkpoints_dir, session)

    def _save_best_checkpoint(self, session):
        self._save_checkpoint(self.best_checkpoints_dir, session)

    def _train_state_exists(self):
        return os.path.exists(self.train_state_path)

    def _load_train_state(self):
        return pickle.load(open(self.train_state_path, "rb"))

    def _save_train_state(self, state):
        pickle.dump(state, open(self.train_state_path, "wb"))

    def train(self):
        with self.graph.as_default():
            # Init the variables
            init = tf.global_variables_initializer()

            max_epochs = 100000
            batch_size = 10
            report_epochs = 25

            with tf.Session() as session:
                if self._train_state_exists():
                    state = self._load_train_state()
                else:
                    state = {
                        "step": 1,
                        "best_valid_loss": sys.float_info.max
                    }

                if self._latest_checkpoint_exists():
                    self._load_latest_checkpoint(session)
                else:
                    session.run(init)

                while state["step"] < max_epochs:
                    batch_samples, batch_labels = self.train_minibatch_loader.random_minibatch()
                    train_loss, train_acc, _, train_summary = session.run([self.loss,
                                                                        self.accuracy,
                                                                        self.optimiser,
                                                                        self.summary_op],
                                                                       feed_dict={self.inputs: batch_samples,
                                                                                  self.true_outputs: batch_labels,
                                                                                  self.is_training: True,
                                                                                  self.input_keepprob: .95,
                                                                                  self.conv_keepprob: .9,
                                                                                  self.fc_keepprob: .6})
                    self.train_writer.add_summary(train_summary, state["step"])
                    self.train_writer.flush()

                    self._save_train_state(state)

                    if state["step"] % report_epochs == 0:
                        print("Epoch " + str(state["step"]))
                        print("Training Loss={:.6f}".format(train_loss))
                        print("Training Accuracy={:.6f}".format(train_acc))
                        for (valid_images, valid_outputs) in self.test_minibatch_loader.sequential_minibatches():
                            valid_loss, valid_acc, valid_summary = session.run(
                                [self.loss, self.accuracy, self.summary_op],
                                feed_dict={self.inputs: valid_images,
                                           self.true_outputs: valid_outputs,
                                           self.is_training: False,
                                           self.input_keepprob: 1,
                                           self.conv_keepprob: 1,
                                           self.fc_keepprob: 1})

                            self.valid_writer.add_summary(valid_summary, state["step"])
                            self.valid_writer.flush()
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

                            if valid_loss > state["best_valid_loss"]:
                                state["best_valid_loss"] = valid_loss
                                self._save_best_checkpoint(session)

                    state["step"] += 1

                print("Training finished!")
                # test_loss, test_acc = session.run([self.loss, self.accuracy],
                #                                feed_dict={self.inputs: test_dataset,
                #                                           self.true_outputs: test_labels,
                #                                           self.is_training: False,
                #                                           self.input_keepprob: 1,
                #                                           self.conv_keepprob: 1,
                #                                           self.fc_keepprob: 1})
                #
                # print("Test Loss=" + "{:.6f}".format(test_loss))
                # print("Test Accuracy=" + "{:.6f}".format(test_acc))

if __name__ == "__main__":
    classifier = ShufflenetClassifier("pascal_voc_multiclassifier",
                                      n_classes=20,
                                      input_resolution=[227, 227])

    classifier.initialize_classifier()
    classifier.train()