import tensorflow as tf

from tensorflow.contrib.slim.nets import resnet_v2

import data
import training
import os
import cPickle
import util
import shutil
import numpy as np
import graph_builder

def analyze_images():
    voc_path = "/media/imanoid/DATA/workspace/data/VOCdevkit/VOC2012"
    img_dir = os.path.join(voc_path, 'JPEGImages')
    ann_dir = os.path.join(voc_path, 'Annotations')
    set_dir = os.path.join(voc_path, 'ImageSets', 'Main')
    pickle_dir = os.path.join(voc_path, 'Pickle')

    input_resolution = (256, 256)

    loader = data.PascalVocLoader(set_dir, ann_dir, img_dir, pickle_dir, image_resolution=input_resolution)

    loader.show_image_per_label()

def train_squeezenet_classifier():
    builder = graph_builder.SqueezeNetBuilder()
    n_outputs = 19
    input_resolution = (256, 256)

    # traindata path
    tensorboard_dir = "/media/imanoid/DATA/workspace/data/tensorboard/squeezenet"
    train_dir = os.path.join(tensorboard_dir, "train")
    test_dir = os.path.join(tensorboard_dir, "test")
    checkpoints_dir = os.path.join(tensorboard_dir, "checkpoints")
    checkpoint_path = os.path.join(checkpoints_dir, "squeezenet.cp")
    state_path = os.path.join(tensorboard_dir, "state.pickle")

    # samples path
    voc_path = "/media/imanoid/DATA/workspace/data/VOCdevkit/VOC2012"
    img_dir = os.path.join(voc_path, 'JPEGImages')
    ann_dir = os.path.join(voc_path, 'Annotations')
    set_dir = os.path.join(voc_path, 'ImageSets', 'Main')
    pickle_dir = os.path.join(voc_path, 'Pickle')

    loader = data.PascalVocLoader(set_dir, ann_dir, img_dir, pickle_dir, image_resolution=input_resolution)
    labels = np.array(loader.get_labels())
    loader.initialize(reset=False)
    test_dataset, test_labels = loader.load_testset()
    valid_dataset, valid_labels = loader.load_validset()

    batch_norm = True

    g = tf.Graph()

    with g.as_default():
        inputs = tf.placeholder(tf.float32, [None, input_resolution[0], input_resolution[1], 3])
        true_outputs = tf.placeholder(tf.float32, [None, n_outputs])

        is_training = tf.placeholder(tf.bool)
        input_keepprob = tf.placeholder(builder.dtype)
        conv_keepprob = tf.placeholder(builder.dtype)
        fc_keepprob = tf.placeholder(builder.dtype)

        root_output = builder.build_root(inputs, input_keepprob=input_keepprob, batch_norm=batch_norm, is_training=is_training)
        segment_tails = builder.build_trunk(root_output, 16, [2, 6, 12, 15], conv_keepprob=conv_keepprob, batch_norm=batch_norm, is_training=is_training)
        outputs = builder.build_classifier_head(segment_tails[-1], n_outputs, fc_keepprob=fc_keepprob)

        logits = tf.reshape(outputs, [-1, n_outputs])

        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=true_outputs)

        loss = tf.reduce_mean(cross_entropy)
        tf.summary.scalar('loss', loss)

        # evaluate model
        correct_pred = tf.equal(tf.argmax(logits, 1), tf.argmax(true_outputs, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
        tf.summary.scalar('accuracy', accuracy)

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with g.control_dependencies(update_ops):
            optimiser = tf.train.AdamOptimizer(learning_rate=1e-3, epsilon=1e-8).minimize(loss)

        # Init the variables
        init = tf.global_variables_initializer()

        # Prepare to save checkpoints
        checkpoint_saver = tf.train.Saver()

        # summaries

        # init dirs
        if not os.path.isdir(train_dir):
            os.makedirs(train_dir)

        if not os.path.isdir(test_dir):
            os.makedirs(test_dir)

        if not os.path.isdir(checkpoints_dir):
            os.makedirs(checkpoints_dir)

        # summary writers
        train_writer = tf.summary.FileWriter(train_dir, graph=tf.get_default_graph())
        test_writer = tf.summary.FileWriter(test_dir, graph=tf.get_default_graph())

        summary_op = tf.summary.merge_all()

        max_epochs = 100000
        batch_size = 10
        report_epochs = 25

        with tf.Session() as sess:
            step = 1

            if os.path.exists(state_path):
                state = cPickle.load(open(state_path, "rb"))
                step = state["step"]

            if os.path.exists(os.path.join(checkpoints_dir, "checkpoint")):
                checkpoint_saver.restore(sess, tf.train.latest_checkpoint(checkpoints_dir))
            else:
                sess.run(init)

            while step < max_epochs:
                batch_samples, batch_labels = loader.load_trainset_random_minibatch(batch_size)
                train_loss, train_acc, _, train_summary, train_logits, train_gt = sess.run([loss, accuracy, optimiser, summary_op, logits, correct_pred],
                                                      feed_dict={inputs: batch_samples,
                                                                 true_outputs: batch_labels,
                                                                 is_training: True,
                                                                 input_keepprob: .95,
                                                                 conv_keepprob: .9,
                                                                 fc_keepprob: .6})



                train_writer.add_summary(train_summary, step)
                train_writer.flush()

                state = {
                    "step": step
                }
                cPickle.dump(state, open(state_path, "wb"))

                if step % report_epochs == 0:
                    test_loss, test_acc, test_summary, test_logits, test_gt = sess.run([loss, accuracy, summary_op, logits, correct_pred],
                                                   feed_dict={inputs: test_dataset,
                                                              true_outputs: test_labels,
                                                              is_training: False,
                                                              input_keepprob: 1,
                                                              conv_keepprob: 1,
                                                              fc_keepprob: 1})

                    print("Epoch " + str(step))
                    print("Training Loss={:.6f}".format(train_loss))
                    print("Training Accuracy={:.6f}".format(train_acc))
                    # pred_labels = labels[np.argmax(train_logits, 1)]
                    # true_labels = labels[np.argmax(batch_labels, 1)]
                    # for i in range(l en(pred_labels)):
                    #     print("%s==%s" % (pred_labels[i], true_labels[i]))
                    # print

                    print("Test Loss={:.6f}".format(test_loss))
                    print("Test Accuracy={:.6f}".format(test_acc))
                    # pred_labels = labels[np.argmax(test_logits, 1)]
                    # true_labels = labels[np.argmax(test_labels, 1)]
                    # for i in range(len(pred_labels)):
                    #     print("%s==%s" % (pred_labels[i], true_labels[i]))
                    # print

                    test_writer.add_summary(test_summary, step)
                    test_writer.flush()

                    checkpoint_saver.save(sess, checkpoint_path)

                step += 1

            print ("Training finished!")
            test_loss, test_acc = sess.run([loss, accuracy],
                                           feed_dict={inputs: valid_dataset,
                                                      true_outputs: valid_labels,
                                                      is_training: False,
                                                      input_keepprob: 1,
                                                      conv_keepprob: 1,
                                                      fc_keepprob: 1})

            print("Validation Loss=" + "{:.6f}".format(test_loss))
            print("Validation Accuracy=" + "{:.6f}".format(test_acc))

if __name__ == "__main__":
    train_squeezenet_classifier()