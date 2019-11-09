import tensorflow as tf
# from swda_data import load_file
import os

from HBLSTM_CRF.config import batchSize
from HBLSTM_CRF.da_model import DAModel
from HBLSTM_CRF.local_utils import pad_sequences, minibatches

os.environ["CUDA_VISIBLE_DEVICES"] = "4,5,6,7"


def main():
    # data, labels = load_file()

    data = [[[1, 2, 3, 4], [1, 2, 3], [2, 3, 5]], [[1, 0], [4]], [[1, 2, 8, 4], [1, 1, 3], [2, 3, 9, 1, 3, 1, 9]],
            [[1, 2, 3, 4, 5, 7, 8, 9], [9, 1, 2, 4], [8, 9, 0, 1, 2]],
            [[1, 2, 4, 3, 2, 3], [9, 8, 7, 5, 5, 5, 5, 5, 5, 5, 5]],
            [[1, 2, 3, 4, 5, 6, 9], [9, 1, 0, 0, 2, 4, 6, 5, 4]],
            [[1, 2, 3, 4, 5, 6, 7, 8, 9], [9, 1, 2, 4], [8, 9, 0, 1, 2]], [[1]],
            [[1, 2, 11, 2, 3, 2, 1, 1, 3, 4, 4], [6, 5, 3, 2, 1, 1, 4, 5, 6, 7], [9, 8, 1], [1, 6, 4, 3, 5, 7, 8],
             [0, 9, 2, 4, 6, 2, 4, 6], [5, 2, 2, 5, 6, 7, 3, 7, 2, 2, 1], [0, 0, 0, 1, 2, 7, 5, 3, 7, 5, 3, 6],
             [1, 3, 6, 6, 3, 3, 3, 5, 6, 7, 2, 4, 2, 1], [1, 2, 4, 5, 2, 3, 1, 5, 1, 1, 2],
             [9, 0, 1, 0, 0, 1, 3, 3, 5, 3, 2], [0, 9, 2, 3, 0, 2, 1, 5, 5, 6], [9, 0, 0, 1, 4, 2, 4, 10, 13, 11, 12],
             [0, 0, 1, 2, 3, 0, 1, 1, 0, 1, 2], [0, 0, 1, 3, 1, 12, 13, 3, 12, 3], [0, 9, 1, 2, 3, 4, 1, 3, 2]]]
    labels = [[1, 2, 1], [0, 3], [1, 2, 1], [1, 0, 2], [2, 1], [1, 1], [2, 1, 2], [4],
              [0, 1, 2, 0, 2, 4, 2, 1, 0, 1, 0, 2, 1, 2, 0]]
    train_data = data[:6]
    train_labels = labels[:6]
    dev_data = data[6:]
    dev_labels = data[6:]
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.4

    with tf.Session(config=config) as sess:
        model = DAModel()
        sess.run(tf.global_variables_initializer())
        clip = 2
        saver = tf.train.Saver()
        # writer = tf.summary.FileWriter("D:\\Experimemts\\tensorflow\\DA\\train", sess.graph)
        writer = tf.summary.FileWriter("train", sess.graph)
        counter = 0
        for epoch in range(100):

            for dialogues, labels in minibatches(train_data, train_labels, batchSize):
                _, dialogue_lengthss = pad_sequences(dialogues, 0)
                word_idss, utterance_lengthss = pad_sequences(dialogues, 0, nlevels=2)
                true_labs = labels
                labs_t, _ = pad_sequences(true_labs, 0)
                counter += 1
                train_loss, train_accuracy, _ = sess.run([model.loss, model.accuracy, model.train_op],
                                                         feed_dict={model.word_ids: word_idss,
                                                                    model.utterance_lengths: utterance_lengthss,
                                                                    model.dialogue_lengths: dialogue_lengthss,
                                                                    model.labels: labs_t, model.clip: clip})
                # writer.add_summary(summary, global_step = counter)
                print("step = {}, train_loss = {}, train_accuracy = {}".format(counter, train_loss, train_accuracy))

                train_precision_summ = tf.Summary()
                train_precision_summ.value.add(
                    tag='train_accuracy', simple_value=train_accuracy)
                writer.add_summary(train_precision_summ, counter)

                train_loss_summ = tf.Summary()
                train_loss_summ.value.add(
                    tag='train_loss', simple_value=train_loss)
                writer.add_summary(train_loss_summ, counter)

                if counter % 1000 == 0:
                    loss_dev = []
                    acc_dev = []
                    for dialogues, labels in minibatches(dev_data, dev_labels, batchSize):
                        _, dialogue_lengthss = pad_sequences(dev_data, 0)
                        word_idss, utterance_lengthss = pad_sequences(dev_data, 0, nlevels=2)
                        true_labs = dev_labels
                        labs_t, _ = pad_sequences(true_labs, 0)
                        dev_loss, dev_accuacy = sess.run([model.loss, model.accuracy],
                                                         feed_dict={model.word_ids: word_idss,
                                                                    model.utterance_lengths: utterance_lengthss,
                                                                    model.dialogue_lengths: dialogue_lengthss,
                                                                    model.labels: labs_t})
                        loss_dev.append(dev_loss)
                        acc_dev.append(dev_accuacy)
                    valid_loss = sum(loss_dev) / len(loss_dev)
                    valid_accuracy = sum(acc_dev) / len(acc_dev)

                    dev_precision_summ = tf.Summary()
                    dev_precision_summ.value.add(
                        tag='dev_accuracy', simple_value=valid_accuracy)
                    writer.add_summary(dev_precision_summ, counter)

                    dev_loss_summ = tf.Summary()
                    dev_loss_summ.value.add(
                        tag='dev_loss', simple_value=valid_loss)
                    writer.add_summary(dev_loss_summ, counter)
                    print("counter = {}, dev_loss = {}, dev_accuacy = {}".format(counter, valid_loss, valid_accuracy))


if __name__ == "__main__":
    main()
