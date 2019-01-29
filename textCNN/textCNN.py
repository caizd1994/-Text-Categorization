import tensorflow as tf


class TextCNN():
    def __init__(self):
        self.num_classes = 2
        self.batch_size = 128
        self.sequence_length = 208
        self.vocab_size = 77205
        self.embed_size = 300
        self.learning_rate = tf.Variable(0.01, trainable=False, name="learning_rate")#ADD learning_rate
        self.learning_rate_decay_half_op = tf.assign(self.learning_rate, self.learning_rate * 0.9)
        self.lr_decay_rate = 0.99
        self.filter_sizes = [3,4,5] # it is a list of int. e.g. [3,4,5]
        self.num_filters = 2
        self.initializer = tf.initializers.random_normal(stddev=0.01)
        self.num_filters_total = self.num_filters * len(self.filter_sizes) #how many filters totally.
        # self.multi_label_flag = multi_label_flag
        # self.clip_gradients = clip_gradients
        self.is_training_flag = tf.placeholder(tf.bool, name="is_training_flag")
        self.input_x = tf.placeholder(tf.int32, [None, self.sequence_length], name="input_x")
        self.input_y = tf.placeholder(tf.float32,[None,self.num_classes], name="input_y_multilabel")  # y:[None,num_classes]. this is for multi-label classification only.
        self.dropout_keep_prob = 0.9
        self.global_step = tf.Variable(0, name="global_step", trainable=False)
        self.is_train = True
        self.enable_lr_decay = False
        self.embedding_placeholder = tf.placeholder(tf.float32, [self.vocab_size, self.embed_size])
        self.Embedding = tf.Variable(tf.constant(0.0, shape=[self.vocab_size, self.embed_size]), trainable=False,
                                     name='word_embedding')
        # self.embedding_placeholder = tf.placeholder(tf.float32, [self.vocab_size, self.embed_size])

        self.W_projection = tf.get_variable("W_projection", shape=[self.num_filters_total, self.num_classes],
                                            initializer=tf.zeros_initializer)  # [embed_size,label_size]
        self.b_projection = tf.get_variable("b_projection", shape=[self.num_classes],initializer=tf.zeros_initializer)  # [label_size] #ADD 2017.06.09

    # def init_embedding_weights(self):
    #     with tf.name_scope("embedding"):

    def cnn_single_layer(self):
        pooled_outputs = []
        for i, filter_size in enumerate(self.filter_sizes):
            with tf.variable_scope("convolution-pooling-%s" % filter_size):
                filter = tf.get_variable("filter_%s"%filter_size,[filter_size,self.embed_size,1,self.num_filters],initializer=tf.glorot_normal_initializer())

                conv = tf.nn.conv2d(self.setence_embeddings,filter,strides=[1,1,1,1],padding="VALID",name='conv')
                conv = tf.layers.batch_normalization(conv, trainable=True)

                b = tf.get_variable("b-%s" % filter_size, [self.num_filters],initializer=tf.zeros_initializer)
                result = tf.nn.relu(tf.nn.bias_add(conv,b),"relu")
                pooled_temp = tf.nn.max_pool(result,ksize=[1,self.sequence_length-filter_size+1,1,1],strides=[1,1,1,1], padding='VALID',name="pool")
                pooled_outputs.append(pooled_temp)
        self.h_pool = tf.concat(pooled_outputs,3)
        self.h_pool_flat = tf.reshape(self.h_pool,[-1,self.num_filters_total])
        # with tf.name_scope("dropout"):
        #     self.h_pool_flat = tf.nn.dropout(self.h_pool_flat,keep_prob=self.dropout_keep_prob)
        h = tf.layers.dense(self.h_pool_flat,self.num_filters_total,activation=tf.nn.tanh,kernel_initializer=tf.glorot_normal_initializer())
        return  h

    def loss(self,l2_lambda=0.0001):#0.001
        with tf.name_scope("loss"):
            #input: `logits`:[batch_size, num_classes], and `labels`:[batch_size]
            #output: A 1-D `Tensor` of length `batch_size` of the same type as `logits` with the softmax cross entropy loss.
            losses = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.input_y, logits=self.logits);#sigmoid_cross_entropy_with_logits.#losses=tf.nn.softmax_cross_entropy_with_logits(labels=self.input_y,logits=self.logits)
            #print("1.sparse_softmax_cross_entropy_with_logits.losses:",losses) # shape=(?,)
            loss=tf.reduce_mean(losses)#print("2.loss.loss:", loss) #shape=()
            l2_losses = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables() if 'bias' not in v.name]) * l2_lambda
            loss=loss+l2_losses
        return loss

    def train_op(self,embedding,features):
        self.embedding_placeholder = embedding
        self.input_x = features['word']
        self.input_y = features['label']
        self.Embedding.assign(self.embedding_placeholder)
        # self.init_embedding_weights()
        self.word_emb = tf.nn.embedding_lookup(self.Embedding,self.input_x)
        self.setence_embeddings = tf.expand_dims(self.word_emb,-1)

        h = self.cnn_single_layer()
        with tf.name_scope("output"):
            self.logits = tf.matmul(h,self.W_projection) + self.b_projection
        loss = self.loss()
        train_softmax = tf.nn.softmax(self.logits)
        train_correct_prediction = tf.equal(tf.argmax(train_softmax, 1), self.input_y)
        train_accuracy = tf.reduce_mean(tf.cast(train_correct_prediction, tf.float32))

        training_loss_summary = tf.summary.scalar("training_loss", loss)
        training_accuracy_summary = tf.summary.scalar("training_accuracy", train_accuracy)
        train_summaries = tf.summary.merge(
            [training_loss_summary, training_accuracy_summary])

        if self.is_train:
            self.global_step = tf.Variable(0, name="global_step", trainable=False)
            if self.enable_lr_decay:
                starter_learning_rate = self.learning_rate
                learning_rate = tf.train.exponential_decay(starter_learning_rate,self.global_step,5000,self.lr_decay_rate,staircase=True)
            else:
                learning_rate = self.learning_rate
            optimizer = tf.train.AdamOptimizer(learning_rate)
            update_op = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_op):
                train_step = optimizer.minimize(loss, global_step=self.global_step)
            tf.get_variable_scope().reuse_variables()

            return train_step,loss, self.global_step,train_accuracy,train_summaries
        else:
            return loss,train_accuracy,train_summaries
if __name__ == '__main__':
    print("hello")