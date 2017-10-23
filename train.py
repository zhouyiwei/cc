import tensorflow as tf
from utils import *
from sklearn.model_selection import KFold
from models import *
import time
import datetime

tf.app.flags.DEFINE_string("dir", "/data", "folder directory")
tf.app.flags.DEFINE_string("training_file", "clickbait17-validation-170630", "Training data file")
tf.app.flags.DEFINE_string("validation_file", "clickbait17-train-170331", "Validation data file")
tf.app.flags.DEFINE_integer("epochs", 20, "epochs")
tf.app.flags.DEFINE_integer("batch_size", 32, "batch_size")
tf.app.flags.DEFINE_string("filter_sizes", "3,4,5", "Comma-separated filter sizes")
tf.app.flags.DEFINE_integer("num_filters", 100, "Number of filters per filter size")
tf.app.flags.DEFINE_float("dropout_rate_hidden", 0.5, "Dropout rate of hidden layer")
tf.app.flags.DEFINE_float("dropout_rate_cell", 0.3, "Dropout rate of rnn cell")
tf.app.flags.DEFINE_float("dropout_rate_embedding", 0.2, "Dropout rate of word embedding")
tf.app.flags.DEFINE_integer("state_size", 64, "state_size")
tf.app.flags.DEFINE_integer("hidden_size", 0, "hidden_size")
tf.app.flags.DEFINE_string("timestamp", "0715", "Timestamp")
tf.app.flags.DEFINE_integer("y_len", 4, "how to interpret the annotation")
tf.app.flags.DEFINE_string("model", "SAN", "which model to use")
tf.app.flags.DEFINE_boolean("use_target_description", False, "whether to use the target description as input")
tf.app.flags.DEFINE_boolean("use_image", False, "whether to use the image as input")
tf.app.flags.DEFINE_float("learning_rate", 0.005, "learning rate")
tf.app.flags.DEFINE_integer("embedding_size", 100, "embedding size")
tf.app.flags.DEFINE_float("gradient_clipping_value", 2, "gradient clipping value")

FLAGS = tf.app.flags.FLAGS


def main(argv=None):
    np.random.seed(81)
    word2id, embedding = load_embeddings(fp=os.path.join(FLAGS.dir, "glove.6B."+str(FLAGS.embedding_size)+"d.txt"), embedding_size=FLAGS.embedding_size)
    with open(os.path.join(FLAGS.dir, 'word2id.json'), 'w') as fout:
        json.dump(word2id, fp=fout)
    # vocab_size = embedding.shape[0]
    # embedding_size = embedding.shape[1]
    ids, post_texts, truth_classes, post_text_lens, truth_means, target_descriptions, target_description_lens, image_features = read_data(word2id=word2id, fps=[os.path.join(FLAGS.dir, FLAGS.training_file), os.path.join(FLAGS.dir, FLAGS.validation_file)], y_len=FLAGS.y_len, use_target_description=FLAGS.use_target_description, use_image=FLAGS.use_image)
    post_texts = np.array(post_texts)
    truth_classes = np.array(truth_classes)
    post_text_lens = np.array(post_text_lens)
    truth_means = np.array(truth_means)
    shuffle_indices = np.random.permutation(np.arange(len(post_texts)))
    post_texts = post_texts[shuffle_indices]
    truth_classes = truth_classes[shuffle_indices]
    post_text_lens = post_text_lens[shuffle_indices]
    truth_means = truth_means[shuffle_indices]
    max_post_text_len = max(post_text_lens)
    print max_post_text_len
    post_texts = pad_sequences(post_texts, max_post_text_len)

    target_descriptions = np.array(target_descriptions)
    target_description_lens = np.array(target_description_lens)
    target_descriptions = target_descriptions[shuffle_indices]
    target_description_lens = target_description_lens[shuffle_indices]
    max_target_description_len = max(target_description_lens)
    print max_target_description_len
    target_descriptions = pad_sequences(target_descriptions, max_target_description_len)

    image_features = np.array(image_features)

    data = np.array(list(zip(post_texts, truth_classes, post_text_lens, truth_means, target_descriptions, target_description_lens, image_features)))
    kf = KFold(n_splits=5)
    round = 1
    val_scores = []
    val_accs = []
    for train, validation in kf.split(data):
        train_data, validation_data = data[train], data[validation]
        g = tf.Graph()
        with g.as_default() as g:
            tf.set_random_seed(81)
            with tf.Session(graph=g) as sess:
                if FLAGS.model == "DAN":
                    model = DAN(x1_maxlen=max_post_text_len, y_len=len(truth_classes[0]), x2_maxlen=max_target_description_len, embedding=embedding, filter_sizes=list(map(int, FLAGS.filter_sizes.split(","))), num_filters=FLAGS.num_filters, hidden_size=FLAGS.hidden_size, state_size=FLAGS.state_size, x3_size=len(image_features[0]))
                if FLAGS.model == "CNN":
                    model = CNN(x1_maxlen=max_post_text_len, y_len=len(truth_classes[0]), x2_maxlen=max_target_description_len, embedding=embedding, filter_sizes=list(map(int, FLAGS.filter_sizes.split(","))), num_filters=FLAGS.num_filters, hidden_size=FLAGS.hidden_size, state_size=FLAGS.state_size, x3_size=len(image_features[0]))
                if FLAGS.model == "BiRNN":
                    model = BiRNN(x1_maxlen=max_post_text_len, y_len=len(truth_classes[0]), x2_maxlen=max_target_description_len, embedding=embedding, filter_sizes=list(map(int, FLAGS.filter_sizes.split(","))), num_filters=FLAGS.num_filters, hidden_size=FLAGS.hidden_size, state_size=FLAGS.state_size, x3_size=len(image_features[0]))
                if FLAGS.model == "SAN":
                    model = SAN(x1_maxlen=max_post_text_len, y_len=len(truth_classes[0]), x2_maxlen=max_target_description_len, embedding=embedding, filter_sizes=list(map(int, FLAGS.filter_sizes.split(","))), num_filters=FLAGS.num_filters, hidden_size=FLAGS.hidden_size, state_size=FLAGS.state_size, x3_size=len(image_features[0]), attention_size=2*FLAGS.state_size)
                global_step = tf.Variable(0, name="global_step", trainable=False)
                optimizer = tf.train.AdamOptimizer(FLAGS.learning_rate)
                grads_and_vars = optimizer.compute_gradients(model.loss)
                if FLAGS.gradient_clipping_value:
                    grads_and_vars = [(tf.clip_by_value(grad, -FLAGS.gradient_clipping_value, FLAGS.gradient_clipping_value), var) for grad, var in grads_and_vars]
                train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

                out_dir = os.path.join(FLAGS.dir, "runs", FLAGS.timestamp)
                # loss_summary = tf.summary.scalar("loss", model.loss)
                # acc_summary = tf.summary.scalar("accuracy", model.accuracy)
                # train_summary_op = tf.summary.merge([loss_summary, acc_summary])
                # train_summary_dir = os.path.join(out_dir, "summaries", "train")
                # train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)
                # val_summary_op = tf.summary.merge([loss_summary, acc_summary])
                # val_summary_dir = os.path.join(out_dir, "summaries", "validation")
                # val_summary_writer = tf.summary.FileWriter(val_summary_dir, sess.graph)

                checkpoint_dir = os.path.join(out_dir, "checkpoints")
                checkpoint_prefix = os.path.join(checkpoint_dir, FLAGS.model+str(round))
                if not os.path.exists(checkpoint_dir):
                    os.makedirs(checkpoint_dir)
                saver = tf.train.Saver()

                sess.run(tf.global_variables_initializer())

                def train_step(input_x1, input_y, input_x1_len, input_z, input_x2, input_x2_len, input_x3):
                    feed_dict = {model.input_x1: input_x1,
                                 model.input_y: input_y,
                                 model.input_x1_len: input_x1_len,
                                 model.input_z: input_z,
                                 model.dropout_rate_hidden: FLAGS.dropout_rate_hidden,
                                 model.dropout_rate_cell: FLAGS.dropout_rate_cell,
                                 model.dropout_rate_embedding: FLAGS.dropout_rate_embedding,
                                 model.batch_size: len(input_x1),
                                 model.input_x2: input_x2,
                                 model.input_x2_len: input_x2_len,
                                 model.input_x3: input_x3}
                    _, step, loss, mse, accuracy = sess.run([train_op, global_step, model.loss, model.mse, model.accuracy], feed_dict)
                    time_str = datetime.datetime.now().isoformat()
                    print("{}: step {}, loss {:g}, mse {:g}, acc {:g}".format(time_str, step, loss, mse, accuracy))
                    # train_summary_writer.add_summary(summaries, step)

                def validation_step(input_x1, input_y, input_x1_len, input_z, input_x2, input_x2_len, input_x3, writer=None):
                    feed_dict = {model.input_x1: input_x1,
                                 model.input_y: input_y,
                                 model.input_x1_len: input_x1_len,
                                 model.input_z: input_z,
                                 model.dropout_rate_hidden: 0,
                                 model.dropout_rate_cell: 0,
                                 model.dropout_rate_embedding: 0,
                                 model.batch_size: len(input_x1),
                                 model.input_x2: input_x2,
                                 model.input_x2_len: input_x2_len,
                                 model.input_x3: input_x3}
                    step, loss, mse, accuracy = sess.run([global_step, model.loss, model.mse, model.accuracy], feed_dict)
                    time_str = datetime.datetime.now().isoformat()
                    print("{}: step {}, loss {:g}, mse {:g}, acc {:g}".format(time_str, step, loss, mse, accuracy))
                    # if writer:
                    #     writer.add_summary(summaries, step)
                    return mse, accuracy

                print("\nValidation: ")
                post_text_val, truth_class_val, post_text_len_val, truth_mean_val,  target_description_val, target_description_len_val, image_feature_val= zip(*validation_data)
                validation_step(post_text_val, truth_class_val, post_text_len_val, truth_mean_val, target_description_val, target_description_len_val, image_feature_val)
                print("\n")
                min_mse_val = np.inf
                acc = np.inf
                for i in range(FLAGS.epochs):
                    batches = get_batch(train_data, FLAGS.batch_size)
                    for batch in batches:
                        post_text_batch, truth_class_batch, post_text_len_batch, truth_mean_batch, target_description_batch, target_description_len_batch, image_feature_batch = zip(*batch)
                        train_step(post_text_batch, truth_class_batch, post_text_len_batch, truth_mean_batch, target_description_batch, target_description_len_batch, image_feature_batch)
                    print("\nValidation: ")
                    mse_val, acc_val = validation_step(post_text_val, truth_class_val, post_text_len_val, truth_mean_val, target_description_val, target_description_len_val, image_feature_val)
                    print("\n")
                    if mse_val < min_mse_val:
                        min_mse_val = mse_val
                        acc = acc_val
                        # saver.save(sess, checkpoint_prefix)
        round += 1
        val_scores.append(min_mse_val)
        val_accs.append(acc)
    print np.mean(val_scores)
    print np.mean(val_accs)


if __name__ == "__main__":
    tf.app.run()

