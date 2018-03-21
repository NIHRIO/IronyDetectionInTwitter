# -*- coding: UTF-8 -*-
import re
import numpy as np
import pickle
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.metrics import f1_score, accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
import os
from sklearn import svm
from nltk.tokenize import TweetTokenizer
import string
from nltk.stem.porter import PorterStemmer
import nltk
import spacy
import emoji
from sklearn.model_selection import cross_val_score, KFold

stemmer = PorterStemmer()
stopset = set(list(string.punctuation))


class DataProcessor(object):
    n_features = 1000
    n_lsi = 100
    random_state = 42
    n_iter = 100
    n_train = 3834
    n_list = [80, 100, 120]

    def __init__(self):
        self.ROOT_DIR = os.path.dirname(__file__)
        self.ROOT_DIR = os.path.abspath(os.path.join(self.ROOT_DIR, os.pardir)) + "/"
        self.embedding_model = None
        self.normalisation_dict = dict()
        self.positive_set = set()
        self.negative_set = set()
        self.tokenizer = TweetTokenizer()
        self.clusters = {}
        self.cluster_word_count = {}
        self.n_clusters = {}
        for n in self.n_list:
            self.load_brown_clusters(n)

    def load_brown_clusters(self, n_cluster):
        self.clusters[n_cluster] = {}
        self.cluster_word_count[n_cluster] = {}
        ids = set()
        id_map = {}
        file_name = self.ROOT_DIR + "data/brownclusters/processed_data-c" + str(n_cluster) + "-p1.out/paths"
        file = open(file_name)
        for line in file:
            elements = line.strip().split("\t")
            if len(elements) == 3:
                cluster_id = int(elements[0], 2)
                word = elements[1]
                freq = int(elements[2])
                before = len(ids)
                ids.add(cluster_id)
                after = len(ids)
                if after > before:
                    id_map[cluster_id] = len(ids) - 1
                cluster_id = id_map[cluster_id]
                if word not in self.clusters[n_cluster]:
                    self.clusters[n_cluster][word] = {}
                self.clusters[n_cluster][word][cluster_id] = freq
                if cluster_id not in self.cluster_word_count[n_cluster]:
                    self.cluster_word_count[n_cluster][cluster_id] = 0
                self.cluster_word_count[n_cluster][cluster_id] += freq
        self.n_clusters[n_cluster] = len(ids)

    def get_brown_cluster_vector(self, tweet, n_cluster):
        tweet = re.split("\\s+", tweet)
        output = np.zeros(self.n_clusters[n_cluster])
        for word in tweet:
            if word in self.clusters[n_cluster]:
                cluster_ids = self.clusters[n_cluster][word]
                for id in cluster_ids:
                    output[id] += cluster_ids[id]
        for id in range(len(output)):
            output[id] = output[id] * 1.0/self.cluster_word_count[n_cluster][id]
        return output

    @staticmethod
    def remove_tweet_tags(tweet_str):
        tweet_str = tweet_str.replace("taggeduser", "").replace("url", "").replace("number", "")
        tweet_str = re.sub("#", " ", tweet_str)
        return re.sub("\\s+", " ", tweet_str).strip()

    @staticmethod
    def normalise_hashtag(hashtag):
        word = ""
        output= []
        for char in hashtag:
            if char.isupper() or char == "#":
                if len(word) > 0:
                    output.append(word)
                    word = ""
            if char != "#":
                word += char


        output.append(word)
        return " ".join(output)

    def extract_pos_tags(self, tweet_str):
        tweet_str = self.remove_tweet_tags(tweet_str)
        tokenizing_text = self.tokenizer.tokenize(tweet_str)
        pos_tags = nltk.pos_tag(tokenizing_text)
        output = [tuple[1] for tuple in pos_tags]
        return np.array(output)

    def process_data(self, train_file, test_file, load_saved_data=True):
        if not train_file.startswith(self.ROOT_DIR):
            train_file = self.ROOT_DIR + train_file

        if not test_file.startswith(self.ROOT_DIR):
            test_file = self.ROOT_DIR + test_file

        saved_training_data_path = self.ROOT_DIR + "data/saved/training_data.pkl"
        saved_test_data_path = self.ROOT_DIR + "data/saved/test_data.pkl"

        if load_saved_data:
            if os.path.exists(saved_training_data_path) and \
                    os.path.exists(saved_test_data_path):
                train_data = self.load_dict(saved_training_data_path)
                test_data = self.load_dict(saved_test_data_path)
                if train_data is not None and test_data is not None:
                    return train_data, test_data

        # for training word brown clusters
        processed_data_file = open(self.ROOT_DIR + "data/processed_data.txt", "w")

        self.load_normalisation_dict()

        self.load_sentiment_words()

        self.embedding_model = spacy.load("en_core_web_md")

        features = []
        labels = []
        text_data = []
        pos_tags = []
        n_train = 0
        train_reader = open(train_file, 'r')
        for line in train_reader:
            elements = line.split("\t")
            elements[2] = self.normalise_tweet(elements[2])
            processed_data_file.write(elements[2].encode("utf8") + "\n")
            if len(elements) == 3 and 'Label' not in elements[1]:
                n_train += 1
                features.append(self.process_a_tweet(elements[2]))
                labels.append(int(elements[1]))
                text_data.append(elements[2])
                pos_tags.append(' '.join(self.extract_pos_tags(elements[2])))

        train_reader.close()

        test_reader = open(test_file, 'r')
        for line in test_reader:
            elements = line.split("\t")
            elements[1] = self.normalise_tweet(elements[1])
            processed_data_file.write(elements[1].encode("utf8") + "\n")
            if len(elements) == 2 and 'tweet index' not in elements[1]:
                features.append(self.process_a_tweet(elements[1]))
                text_data.append(elements[1])
                pos_tags.append(' '.join(self.extract_pos_tags(elements[1])))

        test_reader.close()
        processed_data_file.close()

        # n-gram features for POS tags
        postag_tfidf_vectorizer = TfidfVectorizer(ngram_range=(1, 1), \
                                                  max_features=self.n_features, norm='l2')
        postag_tfidfs = postag_tfidf_vectorizer.fit_transform(pos_tags)
        postag_tfidfs_features = postag_tfidfs.toarray()
        features = np.append(features, postag_tfidfs_features, 1)
        print(len(postag_tfidf_vectorizer.get_feature_names()))
        del postag_tfidfs_features

        # n-gram features for tweets
        # character-based n-grams
        ngram_vectorizer = TfidfVectorizer(analyzer='char_wb', ngram_range=(1, 3), \
                                           max_features=self.n_features, norm='l2')
        counts = ngram_vectorizer.fit_transform(text_data)
        n_grams_features = counts.toarray()
        features = np.append(features, n_grams_features, 1)
        del n_grams_features
        print(len(features[0]))

        # word-based n-grams
        tfidf_vectorizer = TfidfVectorizer(ngram_range=(1, 3), lowercase=False, \
                                           max_features=self.n_features, norm='l2')
        tfidfs = tfidf_vectorizer.fit_transform(text_data)
        tfidfs_features = tfidfs.toarray()
        features = np.append(features, tfidfs_features, 1)
        print(len(features[0]))
        del tfidfs_features

        # LSI features
        if self.n_lsi > 0:
            print("Training LSI!")
            svd_model = TruncatedSVD(n_components=self.n_lsi, \
                                     algorithm='arpack', \
                                     n_iter=self.n_iter, random_state=self.random_state)
            svd_matrix = svd_model.fit_transform(tfidfs)
            features = np.append(features, svd_matrix, 1)
            print(len(features[0]))
            del tfidfs
            del svd_matrix
            print("Got LSI!")

        if not os.path.exists(self.ROOT_DIR + 'data/saved'):
            os.makedirs(self.ROOT_DIR + 'data/saved')
        train_data = {"feature": features[0:n_train], "raw_data": text_data[0:n_train], "label": labels}
        test_data = {"feature": features[n_train:], "raw_data": text_data[n_train:]}
        self.save_dict(train_data, saved_training_data_path)
        self.save_dict(test_data, saved_test_data_path)
        print("Saved data!")
        return train_data, test_data

    def load_normalisation_dict(self):
        reader = open(self.ROOT_DIR + "data/normalisation/emnlp_dict.txt")
        for line in reader:
            elements = re.split("\\s+", line)
            self.normalisation_dict[elements[0].strip()] = elements[1].strip()
        reader.close()
        reader = open(self.ROOT_DIR + "data/normalisation/Test_Set_3802_Pairs.txt")
        for line in reader:
            elements = line.split("\t")[1].split(" | ")
            self.normalisation_dict[elements[0].strip()] = elements[1].strip()

        reader.close()

    def load_sentiment_words(self):
        reader = open(self.ROOT_DIR + "data/sentiment/negative-words.txt")
        for line in reader:
            if len(line.strip()) > 0:
                self.negative_set.add(line.strip())
        reader.close()

        reader = open(self.ROOT_DIR + "data/sentiment/positive-words.txt")
        for line in reader:
            if len(line.strip()) > 0:
                self.positive_set.add(line.strip())
        reader.close()

    @staticmethod
    def save_dict(data, filename):
        with open(filename, 'wb') as f:
            pickle.dump(data, f)
            f.close()

    @staticmethod
    def load_dict(filename):
        with open(filename, 'rb') as f:
            ret_dict = pickle.load(f)
            f.close()
        return ret_dict

    def normalise_tweet(self, tweet_str):

        tweet_str = (emoji.demojize(tweet_str.decode("utf8")))
        tweet_str = re.sub("\\s+", " ", re.sub("http.*?\\s", "url", tweet_str)
                           .replace(":", " ").replace("#", " #").replace("@", " @"))
        tweet = self.tokenizer.tokenize(tweet_str)
        normalised_tweet = ""

        for token_str in tweet:
            normalised_token_str = self.normalise_str(token_str.lower())
            if "haha" in normalised_token_str:
                token_str = "lol"
            if token_str.startswith("@"):
                normalised_tweet += "taggeduser "
            elif token_str.lower().startswith("http"):
                normalised_tweet += "url "
            elif self.is_number(token_str):
                normalised_tweet += "number "
            elif token_str.startswith("#"):
                normalised_tweet += token_str + " "
                normalised_tweet += self.normalise_hashtag(token_str) + " "
            elif normalised_token_str in self.normalisation_dict:
                normalised_tweet += self.normalisation_dict[normalised_token_str] + " "
            else: normalised_tweet += token_str + " "
        return normalised_tweet.strip().lower()

    @staticmethod
    def normalise_str(str_in):
        normalised_str = ""
        count = 0
        pre_char = None
        for i in range(len(str_in)):
            if i > 0:
                if str_in[i] == pre_char:
                    count += 1
                else:
                    count = 0
            if count <= 2:
                normalised_str += str_in[i]
            pre_char = str_in[i]
        return normalised_str

    @staticmethod
    def is_number(s):
        try:
            float(s)
            return True
        except ValueError:
            pass

        try:
            import unicodedata
            unicodedata.numeric(s)
            return True
        except (TypeError, ValueError):
            pass

        return False

    def process_a_tweet(self, tweet_str):
        tweet_vector = []
        if type(tweet_str) != str:
            tweet_str = unicode(tweet_str).encode('utf8')

        tweet_str = tweet_str.decode('utf-8')
        n_token = len(re.split("\\s+", tweet_str.lower()))
        embedding_vector = self.embedding_model(unicode(tweet_str))
        tweet_vector.extend(embedding_vector.vector)
        tweet_vector.append(self.has_irony_hashtag(tweet_str))
        tweet_vector.append(self.get_hash_tag_rate(tweet_str, n_token))
        tweet_vector.append(self.get_tagged_user_rate(tweet_str, n_token))
        tweet_vector.append(self.get_uppercase_rate(tweet_str))
        tweet_vector.extend(self.get_sentiment_word_rate(tweet_str))
        for n in self.n_list:
            tweet_vector.extend(self.get_brown_cluster_vector(tweet_str, n))
        return tweet_vector

    @staticmethod
    # Return the rate of uppercase characters in a tweet
    def get_uppercase_rate(tweet_str):
        count = 0
        for char in tweet_str:
            if char.isupper():
                count = count + 1
        return count * 1.0/len(tweet_str)

    # Return the rates of sentiment words in a tweet
    def get_sentiment_word_rate(self, tweet_str):
        positive_icons = ["grinning face", "beaming face with smiling eyes", "face with tears of joy",
                          "rolling on the floor laughing", "grinning face with big eyes",
                          "grinning face with smiling eyes", "grinning face with sweat", "grinning squinting face",
                          "winking face", "smiling face with smiling eyes", "face savoring food",
                          "smiling face with sunglasses", "smiling face with heart-eyes",
                          "smiling face with heart-shaped eyes", "face blowing a kiss", "kissing face",
                          "kissing face with smiling eyes", "kissing face with closed eyes", "smiling face",
                          "slightly smiling face", "hugging face", "star-struck", ":)", ";)", ":-)", "lol"]
        negative_icons = ["frowning face", "slightly frowning face", "confounded face", "disappointed face",
                          "worried face", "face with steam from nose", "crying face", "loudly crying face",
                          "frowning face with open mouth", "anguished face", "fearful face", "weary face",
                          "exploding head", "grimacing face", "anxious face with sweat", "face screaming in fear",
                          "flushed face", "zany face", "dizzy face", "pouting face",
                          "angry face", "face with symbols on mouth" ,":(", ";(", ":-(", "-.-"]
        sick_icons = ["face with medical mask", "face with thermometer", "face with head-bandage", "nauseated face",
                      "face vomiting", "sneezing face"]
        tweet = re.split("\\s+", tweet_str.lower())
        n_positive_words = 0.0
        n_negative_words = 0.0
        n_not_words = 0.0
        n_pos_icon = 0.0
        n_neg_icon = 0.0
        n_sick_icon = 0.0
        n_icon = 0.0

        for token in tweet:
            if token in self.positive_set:
                n_positive_words += 1
            elif token in self.negative_set:
                n_negative_words += 1
            elif "not" in token or "n't" in token:
                n_not_words += 1
            elif "_" in token:
                icon = token.replace(":", "").replace("_", " ")
                if "smiling" in icon or icon in positive_icons:
                    n_pos_icon += 1
                elif icon in negative_icons:
                    n_neg_icon += 1
                elif icon in sick_icons:
                    n_sick_icon += 1
                else:
                    n_icon += 1
        return n_positive_words/len(tweet), n_negative_words/len(tweet), \
               n_not_words/len(tweet), n_pos_icon/len(tweet),\
               n_neg_icon/len(tweet), n_sick_icon, n_icon/len(tweet)

    def get_tagged_user_rate(self, tweet_str, n_token):
        results = re.findall("@", tweet_str)
        return len(results)*1.0/n_token

    def get_hash_tag_rate(self, tweet_str, n_token):
        results = re.findall("#", tweet_str)
        return len(results) * 1.0 / n_token

    @staticmethod
    def has_irony_hashtag(str):
        regex = re.compile("#not[\W]*$|#not\\s+?#|#not\\s*?\\bhttp\\b|#irony|#sarcasm|#fake|#naah")
        if regex.search(str.lower()):
            return 1.0
        return 0.0

    @staticmethod
    def add_tweet_text(predicted_file, input_file):
        tweet_file = open(input_file, "r")
        tweet_map = {}
        for line in tweet_file:
            if "tweet index" not in line:
                elements = line.split("\t")
                tweet_map[int(elements[0])] = elements[1].strip()
        tweet_file.close()

        label_file = open(predicted_file, "r")
        out_file = open(predicted_file + ".processed", "w")
        for line in label_file:
            elements = line.split("\t")
            id = int(elements[0])
            out_file.write(line.strip()  + "\t" + tweet_map[id] + "\n")
        out_file.close()
        label_file.close()

    @staticmethod
    def lg_predict(train_data, valid_data, test_data, task_name="A"):
        train_features = train_data["feature"]
        train_labels = train_data["label"]
        valid_features = valid_data["feature"]
        valid_labels = valid_data["label"]
        test_features = test_data["feature"]
        lr = LogisticRegression(max_iter=100)
        lr.fit(train_features, train_labels)

        pred_valid_labels = lr.predict(valid_features)
        if task_name == "A":
            f1_valid = f1_score(valid_labels, pred_valid_labels,  pos_label=1)
        else: f1_valid = f1_score(valid_labels, pred_valid_labels,  average="macro")
        print("F1 on valid : %f" % f1_valid)
        return lr.predict(train_features), lr.predict(valid_features), lr.predict(test_features), f1_valid

    @staticmethod
    def rg_predict(train_data, valid_data, test_data, task_name="A"):
        train_features = train_data["feature"]
        train_labels = train_data["label"]
        valid_features = valid_data["feature"]
        valid_labels = valid_data["label"]
        test_features = test_data["feature"]
        rg = RidgeClassifier(max_iter=100)
        rg.fit(train_features, train_labels)
        pred_valid_labels = rg.predict(valid_features)
        if task_name == "A":
            f1_valid = f1_score(valid_labels, pred_valid_labels,  pos_label=1)
        else: f1_valid = f1_score(valid_labels, pred_valid_labels,  average="macro")
        print("F1 on valid : %f" % f1_valid)
        return rg.predict(train_features), rg.predict(valid_features), rg.predict(test_features), f1_valid

    @staticmethod
    def svm_predict(train_data, valid_data, test_data, task_name="A"):
        train_features = train_data["feature"]
        train_labels = train_data["label"]
        valid_features = valid_data["feature"]
        valid_labels = valid_data["label"]
        test_features = test_data["feature"]
        clf = svm.LinearSVC()

        clf.fit(train_features, train_labels)
        pred_valid_labels = clf.predict(valid_features)
        if task_name == "A":
            f1_valid = f1_score(valid_labels, pred_valid_labels,  pos_label=1)
        else: f1_valid = f1_score(valid_labels, pred_valid_labels,  average="macro")
        print("F1 on valid : %f" % f1_valid)
        return clf.predict(train_features), clf.predict(valid_features), clf.predict(test_features), f1_valid

    @staticmethod
    def split_kfolds(data, n_fold):
        kf = KFold(n_splits=n_fold)
        train_data =[]
        valid_data = []
        features = np.array(data["feature"])
        labels = np.array(data["label"])
        raw_data = np.array(data["raw_data"])
        for train_index, valid_index in kf.split(features):
            train = {"feature": features[train_index], "label": labels[train_index],
                     "raw_data": raw_data[train_index]}
            valid = {"feature": features[valid_index], "label": labels[valid_index],
                     "raw_data": raw_data[valid_index]}
            train_data.append(train)
            valid_data.append(valid)
        return train_data, valid_data

