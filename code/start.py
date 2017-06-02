import tensorflow as tf

from tensorflow.contrib.slim.nets import resnet_v2

import data
import training
import os

import util

if __name__ == "__main__":
    n_outputs = 19
    input_resolution = (256, 256)
    voc_path = "/media/imanoid/Data/workspace/data/VOCdevkit/VOC2012"
    img_dir = os.path.join(voc_path, 'JPEGImages')
    ann_dir = os.path.join(voc_path, 'Annotations')
    set_dir = os.path.join(voc_path, 'ImageSets', 'Main')
    pickle_dir = os.path.join(voc_path, 'Pickle')
    loader = data.PascalVocLoader(set_dir, ann_dir, img_dir, pickle_dir, image_resolution=input_resolution)

    loader.initialize(reset=False)
    test_dataset, test_labels = loader.load_testset()
    valid_dataset, valid_labels = loader.load_validset()

    g = tf.Graph()

    with g.as_default():
        inputs = tf.placeholder(tf.float32, [None, input_resolution[0], input_resolution[1], 3])
        true_outputs = tf.placeholder(tf.float32, [None, n_outputs])

        outputs = resnet_v2.resnet_v2_50(inputs, n_outputs,
                                   global_pool=True)

        for element in outputs[1]:
            print(str(element), str(outputs[1][element]))

        logits = tf.reshape(outputs[0], [-1, n_outputs])

        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=true_outputs)

        loss = tf.reduce_mean(cross_entropy)

        # evaluate model
        correct_pred = tf.equal(tf.argmax(logits, 1), tf.argmax(true_outputs, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

        optimiser = tf.train.AdamOptimizer(learning_rate=1e-3, epsilon=1e-8).minimize(loss)

        # Init the variables
        init = tf.global_variables_initializer()

        max_epochs = 100000
        batch_size = 25

        with tf.Session() as sess:
            sess.run(init)
            step = 1

            while step < max_epochs:
                batch_samples, batch_labels = loader.load_trainset_random_minibatch(batch_size)
                [train_loss, train_acc, _] = sess.run(
                    [loss, accuracy, optimiser],
                    feed_dict={inputs: batch_samples,
                               true_outputs: batch_labels})

                if step % 100 == 0:
                    # train_loss, train_acc = sess.run([self.cost, self.accuracy],
                    #                                 feed_dict={self.x: batch_samples,
                    #                                            self.y: batch_labels,
                    #                                            layer1_keep_prob: 1,
                    #                                            layer2_keep_prob: 1,
                    #                                            layer3_keep_prob: 1})

                    pred_y, test_loss, test_acc = sess.run([logits, loss, accuracy],
                                                                    feed_dict={inputs: test_dataset,
                                                                    true_outputs: test_labels})


                    print("Epoch " + str(step))
                    # print(str(pred_y[0, :]))
                    # print(str(batch_labels[0, :]))
                    print("Training Loss={:.6f}".format(train_loss))
                    print("Training Accuracy={:.6f}".format(train_acc))
                    print("Test Loss={:.6f}".format(test_loss))
                    print("Test Accuracy={:.6f}".format(test_acc))

                step += 1

            print ("Training finished!")
            test_loss, test_acc = sess.run([loss, accuracy],
                                           feed_dict={inputs: valid_dataset,
                                                      true_outputs: valid_labels})

            print("Validation Loss=" + "{:.6f}".format(test_loss))
            print("Validation Accuracy=" + "{:.6f}".format(test_acc))


