import numpy as np
import json
import os
import re
import nltk
from gensim.models import Word2Vec
from tweet_utils import *
from collections import Counter
from PIL import Image
import scipy.io
import tensorflow as tf
from scipy import ndimage
import hickle

PAD = "<pad>"  # reserve 0 for pad
UNK = "<unk>"  # reserve 1 for unknown
# tokeniser = nltk.tokenize.stanford.StanfordTokenizer(path_to_jar='./stanford-postagger.jar')
# java_path = "/Library/Java/JavaVirtualMachines/jdk1.8.0_51.jdk/Contents/Home"
# os.environ['JAVAHOME'] = java_path
nltk_tokeniser = nltk.tokenize.TweetTokenizer()
np.random.seed(81)


def process_tweet(text):
    FLAGS = re.MULTILINE | re.DOTALL

    def hashtag(text):
        text = text.group()
        hashtag_body = text[1:]
        result = " ".join(["<hashtag>"] + re.split(r"(?=[A-Z])", hashtag_body, flags=FLAGS))
        return result

    def allcaps(text):
        text = text.group()
        return text.lower() + " <allcaps>"
    eyes = r"[8:=;]"
    nose = r"['`\-]?"

    # function so code less repetitive
    def re_sub(pattern, repl):
        return re.sub(pattern, repl, text, flags=FLAGS)

    text = re_sub(r"https?:\/\/\S+\b|www\.(\w+\.)+\S*", "<url>")
    text = re_sub(r"/"," / ")
    text = re_sub(r"@\w+", "<user>")
    text = re_sub(r"{}{}[)dD]+|[)dD]+{}{}".format(eyes, nose, nose, eyes), "<smile>")
    text = re_sub(r"{}{}p+".format(eyes, nose), "<lolface>")
    text = re_sub(r"{}{}\(+|\)+{}{}".format(eyes, nose, nose, eyes), "<sadface>")
    text = re_sub(r"{}{}[\/|l*]".format(eyes, nose), "<neutralface>")
    text = re_sub(r"<3","<heart>")
    text = re_sub(r"[-+]?[.\d]*[\d]+[:,.\d]*", "<number>")
    text = re_sub(r"#\S+", hashtag)
    text = re_sub(r"([!?.]){2,}", r"\1 <repeat>")
    text = re_sub(r"\b(\S*?)(.)\2{2,}\b", r"\1\2 <elong>")

    ## -- I just don't understand why the Ruby script adds <allcaps> to everything so I limited the selection.
    # text = re_sub(r"([^a-z0-9()<>'`\-]){2,}", allcaps)
    text = re_sub(r"([A-Z]){2,}", allcaps)

    return text


def tokenise(text, with_process=True):
    if with_process:
        return nltk_tokeniser.tokenize(process_tweet(text).lower())
    else:
        # return nltk_tokeniser.tokenize(text)
        return tweet_ark_tokenize(text.lower())


def load_embeddings(fp, embedding_size):
    embedding = []
    vocab = []
    with open(fp, 'r') as f:
        for each_line in f:
            row = each_line.decode('utf-8').split(' ')
            if len(row) == 2:
                continue
            vocab.append(row[0])
            if len(row[1:]) != embedding_size:
                print row[0]
                print len(row[1:])
            embedding.append(np.asarray(row[1:], dtype='float32'))
    word2id = dict(zip(vocab, range(2, len(vocab))))
    word2id[PAD] = 0
    word2id[UNK] = 1
    extra_embedding = [np.zeros(embedding_size), np.random.uniform(-0.1, 0.1, embedding_size)]
    embedding = np.append(extra_embedding, embedding, 0)
    return word2id, embedding


def read_data(fps, word2id=None, y_len=1, use_target_description=False, use_image=False, delete_irregularities=False):
    ids = []
    post_texts = []
    post_text_lens = []
    truth_means = []
    truth_classes = []
    id2truth_class = {}
    id2truth_mean = {}
    target_descriptions = []
    target_description_lens = []
    image_features = []
    num = 0
    for fp in fps:
        if use_image:
            with open(os.path.join(fp, "id2imageidx.json"), "r") as fin:
                id2imageidx = json.load(fin)
            all_image_features = hickle.load(os.path.join(fp, "image_features.hkl"))
        if y_len:
            with open(os.path.join(fp, 'truth.jsonl'), 'rb') as fin:
                for each_line in fin:
                    each_item = json.loads(each_line.decode('utf-8'))
                    if delete_irregularities:
                        if each_item["truthClass"] == "clickbait" and float(each_item["truthMean"]) < 0.5 or each_item["truthClass"] != "clickbait" and float(each_item["truthMean"]) > 0.5:
                            continue
                    if y_len == 4:
                        each_label = [0, 0, 0, 0]
                        for each_key, each_value in Counter(each_item["truthJudgments"]).iteritems():
                            each_label[int(each_key//0.3)] = float(each_value)/5
                        id2truth_class[each_item["id"]] = each_label
                        if each_item["truthClass"] != "clickbait":
                            assert each_label[0]+each_label[1] > each_label[2]+each_label[3]
                        else:
                            assert each_label[0]+each_label[1] < each_label[2]+each_label[3]
                    if y_len == 2:
                        if each_item["truthClass"] == "clickbait":
                            id2truth_class[each_item["id"]] = [1, 0]
                        else:
                            id2truth_class[each_item["id"]] = [0, 1]
                    if y_len == 1:
                        if each_item["truthClass"] == "clickbait":
                            id2truth_class[each_item["id"]] = [1]
                        else:
                            id2truth_class[each_item["id"]] = [0]
                    id2truth_mean[each_item["id"]] = [float(each_item["truthMean"])]
        with open(os.path.join(fp, 'instances.jsonl'), 'rb') as fin:
            for each_line in fin:
                each_item = json.loads(each_line.decode('utf-8'))
                if each_item["id"] not in id2truth_class and y_len:
                    num += 1
                    continue
                ids.append(each_item["id"])
                each_post_text = " ".join(each_item["postText"])
                each_target_description = each_item["targetTitle"]
                if y_len:
                    truth_means.append(id2truth_mean[each_item["id"]])
                    truth_classes.append(id2truth_class[each_item["id"]])
                if word2id:
                    if (each_post_text+" ").isspace():
                        post_texts.append([0])
                        post_text_lens.append(1)
                    else:
                        each_post_tokens = tokenise(each_post_text)
                        post_texts.append([word2id.get(each_token, 1) for each_token in each_post_tokens])
                        post_text_lens.append(len(each_post_tokens))
                else:
                    post_texts.append([each_post_text])
                if use_target_description:
                    if word2id:
                        if (each_target_description+" ").isspace():
                            target_descriptions.append([0])
                            target_description_lens.append(1)
                        else:
                            each_target_description_tokens = tokenise(each_target_description)
                            target_descriptions.append([word2id.get(each_token, 1) for each_token in each_target_description_tokens])
                            target_description_lens.append(len(each_target_description_tokens))
                    else:
                        target_descriptions.append([each_target_description])
                else:
                    target_descriptions.append([])
                    target_description_lens.append(0)
                if use_image:
                    image_features.append(all_image_features[id2imageidx[each_item["id"]]].flatten())
                else:
                    image_features.append([])
    print "Deleted number of items: " + str(num)
    return ids, post_texts, truth_classes, post_text_lens, truth_means, target_descriptions, target_description_lens, image_features


def pad_sequences(sequences, maxlen):
    if maxlen <= 0:
        return sequences
    shape = (len(sequences), maxlen)
    padded_sequences = np.full(shape, 0)
    for i, each_sequence in enumerate(sequences):
        if len(each_sequence) > maxlen:
            padded_sequences[i] = each_sequence[:maxlen]
        else:
            padded_sequences[i, :len(each_sequence)] = each_sequence
    return padded_sequences


def get_batch(data, batch_size, shuffle=True):
    data = np.array(data)
    data_size = len(data)
    batch_num_per_epoch = int((data_size-1)/batch_size)+1
    if shuffle:
        shuffle_indices = np.random.permutation(np.arange(data_size))
        shuffled_data = data[shuffle_indices]
    else:
        shuffled_data = data
    for i in range(batch_num_per_epoch):
        start_ix = i * batch_size
        end_ix = min((i+1)*batch_size, data_size)
        yield shuffled_data[start_ix:end_ix]


def generate_embeddings(fp):
    sentences = []
    files = ["/data/clickbait17-train-170331", "/data/clickbait17-validation-170630", "/data/clickbait17-unlabeled-170429"]
    for each_fp in files:
        with open(os.path.join(each_fp, 'instances.jsonl'), 'rb') as f:
            for each_line in f:
                each_item = json.loads(each_line.decode('utf-8'))
                for each_sentence in each_item["postText"]:
                    sentences.append(tokenise(each_sentence))
                if each_item["targetTitle"]:
                    sentences.append(tokenise(each_item["targetTitle"]))
                if each_item["targetDescription"]:
                    sentences.append(tokenise(each_item["targetDescription"]))
                for each_sentence in each_item["targetParagraphs"]:
                    sentences.append(tokenise(each_sentence))
                for each_sentence in each_item["targetCaptions"]:
                    sentences.append(tokenise(each_sentence))
    word2vec_model = Word2Vec(sentences)
    word2vec_model.wv.save_word2vec_format(os.path.join(fp, "s_clickbait.100.txt"), binary=False)


def extract_vgg_info(vgg_path):
    vgg_data = scipy.io.loadmat(vgg_path)
    normalization_matrix = vgg_data['normalization'][0][0][0]
    mat_mean = np.mean(normalization_matrix, axis=(0,1))
    network_weights = vgg_data["layers"][0]
    return mat_mean, network_weights


def process_image(image_path, mat_mean):
    image = scipy.misc.imread(image_path, mode="RGB")
    image = scipy.misc.imresize(image, [224, 224])
    return image - mat_mean


vgg_layers = ['conv1_1', 'relu1_1',
              'conv1_2', 'relu1_2', 'pool1',
              'conv2_1', 'relu2_1',
              'conv2_2', 'relu2_2', 'pool2',
              'conv3_1', 'relu3_1',
              'conv3_2', 'relu3_2',
              'conv3_3', 'relu3_3',
              'conv3_4', 'relu3_4', 'pool3',
              'conv4_1', 'relu4_1',
              'conv4_2', 'relu4_2',
              'conv4_3', 'relu4_3',
              'conv4_4', 'relu4_4', 'pool4',
              'conv5_1', 'relu5_1',
              'conv5_2', 'relu5_2',
              'conv5_3', 'relu5_3',
              'conv5_4', 'relu5_4']


class VGG19(object):
    def __init__(self, network_weights):
        self.images = tf.placeholder(tf.float32, [None, 224, 224, 3], "images")
        with tf.variable_scope("image_encoder"):
            for i, layer in enumerate(vgg_layers):
                layer_type = layer[:4]
                if layer_type == "conv":
                    weights, bias = network_weights[i][0][0][0][0]
                    weights = np.transpose(weights, (1, 0, 2, 3))
                    bias = bias.reshape(-1)
                    if layer == "conv1_1":
                        h = self.images
                    h = tf.nn.bias_add(tf.nn.conv2d(h, tf.constant(weights), strides=[1, 1, 1, 1], padding="SAME"), bias)
                elif layer_type == "relu":
                    h = tf.nn.relu(h)
                elif layer_type == "pool":
                    h = tf.nn.max_pool(h, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
                if layer == "conv5_3":
                    self.features = tf.reshape(h, [-1, 196, 512])


def extract_image_features(fp="/data/clickbait17-train-170331"):
    # # generate an blank image for no image occasions
    # image = Image.new("RGB", (224, 224))
    # image.save(os.path.join(fp, "media", "_.png"), "PNG")
    id2imageidx = {}
    image_names = [f for f in os.listdir(os.path.join(fp, "media")) if os.path.isfile(os.path.join(fp, "media", f))]
    with open(os.path.join(fp, 'instances.jsonl'), 'rb') as f:
        for each_line in f:
            each_item = json.loads(each_line.decode('utf-8'))
            if each_item["postMedia"]:
                id2imageidx[each_item["id"]] = image_names.index(each_item["postMedia"][0].split("/")[1])+1  # index 0 reserved for no image
            else:
                id2imageidx[each_item["id"]] = 0
    with open(os.path.join(fp, 'id2imageidx.json'), 'w') as fout:
        json.dump(id2imageidx, fp=fout)
    batch_size = 100
    n_examples = len(image_names)
    all_image_features = np.ndarray([n_examples+1, 196, 512], dtype=np.float32)
    all_image_features[0, :] = np.random.uniform(-0.1, 0.1, [196, 512])
    mat_mean, network_weights = extract_vgg_info("/data/imagenet-vgg-verydeep-19.mat")
    vggnet = VGG19(network_weights)
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        for start, end in zip(range(0, n_examples, batch_size), range(batch_size, n_examples+batch_size, batch_size)):
            image_name_batch = image_names[start:end]
            image_batch = np.array(map(lambda f: process_image(os.path.join(fp, "media", f), mat_mean), image_name_batch)).astype(np.float32)
            image_features_batch = sess.run(vggnet.features, feed_dict={vggnet.images: image_batch})
            all_image_features[start+1:end+1, :] = image_features_batch
    hickle.dump(all_image_features, os.path.join(fp, "image_features.hkl"))


if __name__ == '__main__':
    # text = "I TEST alllll kinds of #hashtags and #HASHTAGS, @mentions and 3000 (http://t.co/dkfjkdf). w/ <3 :) haha!!!!!"
    # print(tokenise(text, True))
    # read_data(fp="/data/clickbait17-validation-170630", y_len=4)
    extract_image_features("/data/clickbait17-validation-170630")