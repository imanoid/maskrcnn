import tensorflow as tf
import data
import os
import cPickle
import numpy as np
import graph_builder
import sys


def analyze_images():
    voc_path = "/media/imanoid/DATA/workspace/data/VOCdevkit/VOC2012"
    img_dir = os.path.join(voc_path, 'JPEGImages')
    ann_dir = os.path.join(voc_path, 'Annotations')
    set_dir = os.path.join(voc_path, 'ImageSets', 'Main')
    pickle_dir = os.path.join(voc_path, 'Pickle')

    input_resolution = (192, 192)

    loader = data.PascalVocLoader(set_dir, ann_dir, img_dir, pickle_dir, image_resolution=input_resolution)

    loader.show_images_of_label("person")


def train_squeezenet_classifier():
    builder = graph_builder.SqueezeNetBuilder()
    n_outputs = 2
    input_resolution = (192, 192)

    # traindata path
    tensorboard_dir = "/media/imanoid/DATA/workspace/data/tensorboard/squeezenet"
    train_dir = os.path.join(tensorboard_dir, "train")
    valid_dir = os.path.join(tensorboard_dir, "valid")

    latest_checkpoints_dir = os.path.join(tensorboard_dir, "latest_checkpoints")
    latest_checkpoint_path = os.path.join(latest_checkpoints_dir, "squeezenet.cp")

    best_checkpoints_dir = os.path.join(tensorboard_dir, "best_checkpoints")
    best_checkpoint_path = os.path.join(best_checkpoints_dir, "squeezenet.cp")

    state_path = os.path.join(tensorboard_dir, "state.pickle")

    # samples path
    voc_path = "/media/imanoid/DATA/workspace/data/VOCdevkit/VOC2012"
    img_dir = os.path.join(voc_path, 'JPEGImages')
    ann_dir = os.path.join(voc_path, 'Annotations')
    set_dir = os.path.join(voc_path, 'ImageSets', 'Main')
    pickle_dir = os.path.join(voc_path, 'Pickle')

    loader = data.PascalVocLoader(set_dir, ann_dir, img_dir, pickle_dir, image_resolution=input_resolution, single_label="person")
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

        root_output = builder.add_root(inputs, input_keepprob=input_keepprob, batch_norm=batch_norm,
                                       is_training=is_training)
        segment_tails = builder.add_trunk(root_output, 10, [4, 8], conv_keepprob=conv_keepprob,
                                          batch_norm=batch_norm, is_training=is_training,
                                          squeeze_ratio=0.125)
        outputs = builder.add_classifier_head(segment_tails[-1], n_outputs, fc_keepprob=fc_keepprob,
                                              batch_norm=batch_norm, is_training=is_training)

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

        summary_op = tf.summary.merge_all()

        max_epochs = 100000
        batch_size = 10
        report_epochs = 25

        best_valid_loss = sys.float_info.max

        with tf.Session() as sess:
            step = 1

            if os.path.exists(state_path):
                state = cPickle.load(open(state_path, "rb"))
                step = state["step"]

            if os.path.exists(os.path.join(latest_checkpoints_dir, "checkpoint")):
                latest_checkpoint_saver.restore(sess, tf.train.latest_checkpoint(latest_checkpoints_dir))
            else:
                sess.run(init)

            while step < max_epochs:
                batch_samples, batch_labels = loader.load_trainset_random_minibatch(batch_size)
                train_loss, train_acc, _, train_summary = sess.run([loss, accuracy, optimiser, summary_op],
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
                    valid_loss, valid_acc, valid_summary, valid_logits, valid_labels = sess.run([loss, accuracy, summary_op, logits, true_outputs],
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
                    pred_labels = np.argmax(valid_logits, 1)
                    true_labels = np.argmax(valid_labels, 1)
                    for i in range(len(pred_labels)):
                        print("%s==%s" % (pred_labels[i], true_labels[i]))
                    print

                    valid_writer.add_summary(valid_summary, step)
                    valid_writer.flush()

                    latest_checkpoint_saver.save(sess, latest_checkpoint_path)

                    if valid_loss > best_valid_loss:
                        best_valid_loss = valid_loss
                        best_checkpoint_saver.save(sess, best_checkpoint_path)

                step += 1

            print ("Training finished!")
            test_loss, test_acc = sess.run([loss, accuracy],
                                           feed_dict={inputs: test_dataset,
                                                      true_outputs: test_labels,
                                                      is_training: False,
                                                      input_keepprob: 1,
                                                      conv_keepprob: 1,
                                                      fc_keepprob: 1})

            print("Testation Loss=" + "{:.6f}".format(test_loss))
            print("Testation Accuracy=" + "{:.6f}".format(test_acc))


if __name__ == "__main__":
    train_squeezenet_classifier()
