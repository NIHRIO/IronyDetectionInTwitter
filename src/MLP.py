import logging
import random
import tensorflow as tf
import numpy as np
from sklearn.metrics import f1_score


class MLP(object):
    rd = random.Random()
    logger = logging.getLogger()
    logger.disabled = True
    # Hyper Parameters
    target_size = 4
    hidden_size = 800
    hidden_size_1 = 300
    train_keep_prob = 0.95
    l2_alpha = 0.00001
    learn_rate = 0.0001
    clip_ratio = 5
    batch_size_train = 500
    epochs = 100
    early_stopping = 30

    def calculate_f1(self, session, features_pl, keep_prob_pl, predict, features, labels, task_name="A"):
        pred_labels = self.predict_labels(features, features_pl, keep_prob_pl, predict, session)
        if task_name == "A":
            return f1_score(pred_labels, labels, pos_label=1)
        return f1_score(pred_labels, labels, average='macro')

    def predict_labels(self, features, features_pl, keep_prob_pl, predict, session):
        feed_dict = {features_pl: features, keep_prob_pl: 1.0}
        pred_labels = session.run(predict, feed_dict=feed_dict)
        return pred_labels

    def normalise_label(self, labels):
        normalised_labels = []
        for label in labels:
            normalised_label = np.zeros(self.target_size)
            normalised_label[label] = 1
            normalised_labels.append(normalised_label)
        return np.array(normalised_labels)

    def predict(self, train_data, valid_data, test_data, task_name="A"):
        test_features = np.asarray(test_data["feature"])
        train_features = np.asarray(train_data["feature"])
        pre_train_labels = np.asarray(train_data["label"])
        train_labels = self.normalise_label(pre_train_labels)

        valid_features = np.asarray(valid_data["feature"])
        valid_labels = np.asarray(valid_data["label"])

        n_train = len(train_features)
        feature_size = len(train_features[0])

        # Create placeholders
        features_pl = tf.placeholder(tf.float32, [None, feature_size], name='features')
        labels_pl = tf.placeholder(tf.int32, [None, self.target_size], 'labels')

        keep_prob_pl = tf.placeholder(tf.float32)

        # Infer batch size
        batch_size = tf.shape(features_pl)[0]

        # Define multi-layer perceptron
        hidden_layer = tf.nn.dropout(tf.nn.relu(tf.contrib.layers.linear(features_pl, self.hidden_size)),
                                     keep_prob=keep_prob_pl)
        hidden_layer_1 = tf.nn.dropout(tf.nn.relu(tf.contrib.layers.linear(hidden_layer, self.hidden_size_1)),
                                       keep_prob=keep_prob_pl)

        logits_flat = tf.nn.dropout(tf.contrib.layers.linear(hidden_layer_1, self.target_size), keep_prob=keep_prob_pl)
        logits = tf.reshape(logits_flat, [batch_size, self.target_size])

        # Define L2 loss
        tf_vars = tf.trainable_variables()
        l2_loss = tf.add_n([tf.nn.l2_loss(v) for v in tf_vars if 'bias' not in v.name]) * self.l2_alpha

        # Define overall loss
        # sparse_softmax_cross_entropy_with_logits
        loss = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels_pl) + l2_loss)

        # Define prediction
        softmaxed_logits = tf.nn.softmax(logits)
        predict = tf.argmax(softmaxed_logits, 1)
        # predict = softmaxed_logits

        # Define optimiser
        opt_func = tf.train.AdamOptimizer(self.learn_rate)
        grads, _ = tf.clip_by_global_norm(tf.gradients(loss, tf_vars), self.clip_ratio)
        opt_op = opt_func.apply_gradients(zip(grads, tf_vars))

        with tf.device('/cpu:0'):
            # Perform training
            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            best_valid = 0
            best_train = 0
            stopping = 0
            with tf.Session(config=config) as sess:
                sess.run(tf.global_variables_initializer())
                for epoch in range(self.epochs):
                    total_loss = 0
                    indices = list(range(n_train))
                    self.rd.shuffle(indices)

                    for i in range(n_train // self.batch_size_train):
                        # print(i)
                        batch_indices = indices[i * self.batch_size_train: (i + 1) * self.batch_size_train]
                        batch_features = [train_features[i] for i in batch_indices]
                        batch_labels = [train_labels[i] for i in batch_indices]
                        # print batch_labels
                        batch_feed_dict = {features_pl: batch_features, labels_pl: batch_labels,
                                           keep_prob_pl: self.train_keep_prob}
                        _, current_loss = sess.run([opt_op, loss], feed_dict=batch_feed_dict)
                        total_loss += current_loss

                    valid_f1 = self.calculate_f1(sess, features_pl, keep_prob_pl, predict, valid_features, valid_labels, task_name)
                    train_f1 = self.calculate_f1(sess, features_pl, keep_prob_pl, predict, train_features, pre_train_labels, task_name)

                    if valid_f1 > best_valid:
                        best_valid = valid_f1
                        best_train = train_f1
                        stopping = 0
                    else:
                        stopping += 1

                    if stopping >= self.early_stopping:
                        print("early stop at epoch %d" % epoch)
                        break

                print("Done with F1 on Validation: %f and on Train: %f " % (best_valid, best_train))
                return self.predict_labels(train_features, features_pl, keep_prob_pl, predict, sess), \
                       self.predict_labels(valid_features, features_pl, keep_prob_pl, predict, sess), \
                       self.predict_labels(test_features, features_pl, keep_prob_pl, predict, sess), best_valid

    @staticmethod
    def analyse(data, pred_labels):
        labels = data["label"]
        raw_data = data["raw_data"]
        for i in range(len(labels)):
            if labels[i] != pred_labels[i]:
                print ("%s %d %d" % (raw_data[i], labels[i], pred_labels[i]))
