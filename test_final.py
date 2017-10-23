import tensorflow as tf
from utils import *
from sklearn.model_selection import KFold
# from models import *
import time
import datetime
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import accuracy_score as acc
import argparse

tf.app.flags.DEFINE_string("dir", "/data", "folder directory")
tf.app.flags.DEFINE_string("test_file", "clickbait17-train-170331", "Test data file")
tf.app.flags.DEFINE_string("timestamp", "0715", "Timestamp")
tf.app.flags.DEFINE_integer("max_post_text_len", 39, "Max length of the post text")
tf.app.flags.DEFINE_integer("max_target_description_len", 0, "Max length of the target description")
tf.app.flags.DEFINE_integer("if_annotated", 0, ">=1 if the Test data come with the annotations, 0 otherwise")
tf.app.flags.DEFINE_string("model", "SAN", "which model to use")
tf.app.flags.DEFINE_boolean("use_target_description", False, "whether to use the target description as input")
tf.app.flags.DEFINE_boolean("use_image", False, "whether to use the image as input")
FLAGS = tf.app.flags.FLAGS


def distribution2label(ar):
    ar = np.array(ar)
    constant = np.array([1, 0, 1, 0, 0, 1, 0, 1]).reshape((4, 2))
    ar = np.argmax(np.dot(ar, constant), axis=1)
    return ar


def main(argv=None):
    if not os.path.exists(os.path.join(FLAGS.dir, 'word2id.json')):
        print "Error: no word2id file!"
        return
    if not os.path.exists(os.path.join(FLAGS.dir, "runs", FLAGS.timestamp, "checkpoints")):
        print "Error: no saved model!"
        return
    if FLAGS.use_image and not os.path.exists(os.path.join(FLAGS.dir, FLAGS.test_file, "id2imageidx.json")):
        print "Error: no processed image features!"
        return
    with open(os.path.join(FLAGS.dir, 'word2id.json'), 'r') as fin:
        word2id = json.load(fin)
    ids, post_texts, truth_classes, post_text_lens, truth_means, target_descriptions, target_description_lens, image_features = read_data(word2id=word2id, fps=[argv[1]], y_len=FLAGS.if_annotated, use_target_description=FLAGS.use_target_description, use_image=FLAGS.use_image)
    post_texts = np.array(post_texts)
    truth_classes = np.array(truth_classes)
    post_text_lens = [each_len if each_len <= FLAGS.max_post_text_len else FLAGS.max_post_text_len for each_len in post_text_lens]
    post_text_lens = np.array(post_text_lens)
    truth_means = np.array(truth_means)
    truth_means = np.ravel(truth_means).astype(np.float32)
    post_texts = pad_sequences(post_texts, FLAGS.max_post_text_len)

    if not FLAGS.use_target_description:
        FLAGS.max_target_description_len = 0
    target_descriptions = np.array(target_descriptions)
    target_description_lens = [each_len if each_len <= FLAGS.max_target_description_len else FLAGS.max_target_description_len for each_len in target_description_lens]
    target_description_lens = np.array(target_description_lens)
    target_descriptions = pad_sequences(target_descriptions, FLAGS.max_target_description_len)

    image_features = np.array(image_features)

    all_prediction = []
    all_distribution = []
    for i in range(1, 6):
        tf.reset_default_graph()
        saver = tf.train.import_meta_graph(os.path.join(FLAGS.dir, "runs", FLAGS.timestamp, "checkpoints", FLAGS.model+str(i)+".meta"), clear_devices=True)
        with tf.Session() as sess:
            saver.restore(sess, os.path.join(FLAGS.dir, "runs", FLAGS.timestamp, "checkpoints", FLAGS.model+str(i)))
            g = tf.get_default_graph()
            input_x1 = g.get_tensor_by_name("post_text:0")
            input_x1_len = g.get_tensor_by_name("post_text_len:0")
            dropout_rate_hidden = g.get_tensor_by_name("dropout_rate_hidden:0")
            dropout_rate_cell = g.get_tensor_by_name("dropout_rate_cell:0")
            dropout_rate_embedding = g.get_tensor_by_name("dropout_rate_embedding:0")
            batch_size = g.get_tensor_by_name("batch_size:0")
            input_x2 = g.get_tensor_by_name("target_description:0")
            input_x2_len = g.get_tensor_by_name("target_description_len:0")
            input_x3 = g.get_tensor_by_name("image_feature:0")
            output_prediction = g.get_tensor_by_name("prediction:0")
            output_distribution = g.get_tensor_by_name("distribution:0")
            feed_dict = {input_x1: post_texts,
                         input_x1_len: post_text_lens,
                         dropout_rate_hidden: 0,
                         dropout_rate_cell: 0,
                         dropout_rate_embedding: 0,
                         batch_size: len(post_texts),
                         input_x2: target_descriptions,
                         input_x2_len: target_description_lens,
                         input_x3: image_features}
            prediction, distribution = sess.run([output_prediction, output_distribution], feed_dict)
            prediction = np.ravel(prediction).astype(np.float32)
            all_prediction.append(prediction)
            all_distribution.append(distribution)
            if FLAGS.if_annotated:
                print mse(prediction, truth_means)
                print acc(distribution2label(truth_classes), distribution2label(distribution))
    avg_prediction = np.mean(all_prediction, axis=0)
    avg_distribution = np.mean(all_distribution, axis=0)
    if FLAGS.if_annotated:
        print mse(avg_prediction, truth_means)
        print acc(distribution2label(truth_classes), distribution2label(avg_distribution))
    if not os.path.exists(argv[2]):
        os.makedirs(argv[2])
    with open(os.path.join(argv[2], "predictions.jsonl"), 'w') as output:
        for i in range(len(ids)):
            output.write(json.dumps({"id": ids[i], "clickbaitScore": float(avg_prediction[i])})+'\n')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', dest="input_directory")
    parser.add_argument('-o', dest="output_directory")
    argv = parser.parse_args()
    tf.app.run(argv=[None, argv.input_directory, argv.output_directory])