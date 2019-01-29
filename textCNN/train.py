import tensorflow as tf
import numpy as np
from textCNN.textCNN  import TextCNN
import pickle
import sys
def input_fn(filenames, num_epochs, batch_size):
    def parse_instance(example_proto):
        features = {
            "word":  tf.FixedLenFeature([208],tf.int64),
            "label": tf.FixedLenFeature([],tf.int64),
        }
        parsed_features = tf.parse_example(example_proto, features)
        return parsed_features
    filenames = tf.data.Dataset.from_tensor_slices(filenames)
    dataset = filenames.apply(tf.contrib.data.parallel_interleave(
        lambda filename: tf.data.TFRecordDataset(filename, buffer_size=256 * 1024 * 1024), cycle_length=20
    )
    )
    dataset = dataset \
        .shuffle(buffer_size=batch_size*2) \
        .batch(batch_size) \
        .prefetch(batch_size * 1) \
        .map(parse_instance, num_parallel_calls=30)
    next_element = dataset.repeat(num_epochs).make_one_shot_iterator().get_next()
    return next_element


if __name__ == '__main__':
    dirs = ["train.tfrecords"]
    TENSORBORD_PATH = "../TENSORBORD_PATH"
    CHECKPOINT_FILE = "../CHECKPOINT_FILE"
    train_next = input_fn(dirs, num_epochs=1, batch_size=128)
    steps_to_validate = 100
    sys.stdout.flush()
    with open('remap.pkl', 'rb') as f:
        embedding = pickle.load(f)
    sys.stdout.flush()
    print("embedding_len", len(embedding))
    emb = sorted(embedding, key=lambda x: x[0])
    embedding = np.asarray([x[1] for x in emb])
    print("embedding_shape:",embedding.shape)
    sys.stdout.flush()
    model = TextCNN()
    init_op = [tf.global_variables_initializer(), tf.local_variables_initializer(), tf.tables_initializer()]
    train_logit = model.train_op(embedding,train_next)
    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(init_op)
        sess.run(tf.initialize_all_variables())
        merged = tf.summary.merge_all()
        writer = tf.summary.FileWriter(TENSORBORD_PATH, graph=tf.get_default_graph())
        try:
            while True:
                train_step, loss, global_step, accuracy, train_summary = sess.run(train_logit)
                writer.add_summary(train_summary, global_step=int(global_step))
                if global_step % steps_to_validate == 0:
                    # valid_loss, valid_acc, valid_auc, valid_summary, valid_recall = sess.run(valid_logit)
                    print("step:" + str(global_step) + "\ttrain loss:" + str(loss) + "\ttrain accuracy:" + str(
                        accuracy))
                        #   + "\tvalid loss:" + str(
                        # valid_loss)
                        #   + "\tvalid accuracy:" + str(valid_acc) + "\tvalid auc:" + str(valid_auc) + "\tvalid recall:" + str(
                        # valid_recall))
        except tf.errors.OutOfRangeError:
            saver.save(sess, CHECKPOINT_FILE)
            print("Done")