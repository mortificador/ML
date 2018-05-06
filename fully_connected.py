import tensorflow as tf

n_inputs = 28*28 #MNIST images are 28x28
n_hidden1 = 300
n_hidden2 = 100
n_outputs = 10

X = tf.placeholder(tf.float32, shape=(None, n_inputs), name="X")
y = tf.placeholder(tf.int64, shape=(None), name="y")
training = tf.placeholder_with_default(False, shape=(), name="training")

with tf.name_scope("dnn"):
    hidden1 = tf.layers.dense(X, n_hidden1, name="hidden1")
    bn1 = tf.layers.batch_normalization(hidden1, training=training, momentum=0.9)
    bn1_act = tf.nn.elu(bn1)
    hidden2 = tf.layers.dense(bn1_act, n_hidden2, name="hidden2")
    bn2 = tf.layers.batch_normalization(hidden2, training=training, momentum=0.9)
    bn2_act = tf.nn.elu(bn2)
    logits_before_bn = tf.layers.dense(bn2_act, n_outputs, name="outputs") #output before going through softmax activation
    logits = tf.layers.batch_normalization(logits_before_bn, training=training, momentum=0.9)

with tf.name_scope("loss"):
    #https://www.quora.com/What-is-the-intuition-behind-SoftMax-function
    xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)
    loss = tf.reduce_mean(xentropy, name="loss")

learning_rate = 0.01
with tf.name_scope("train"):
    optimizer = tf.train.MomentumOptimizer(learning_rate, momentum=0.9, use_nesterov=True)
    training_op = optimizer.minimize(loss)

with tf.name_scope("eval"):
    correct = tf.nn.in_top_k(logits, y, 1)
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

init = tf.global_variables_initializer()
saver = tf.train.Saver()

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data")

n_epochs = 40
batch_size = 50
print("llego aqui")
with tf.Session() as sess:
    init.run()
    for epoch in range(n_epochs):
        for iteration in range(mnist.train.num_examples // batch_size):
            X_batch, y_batch = mnist.train.next_batch(batch_size)
            sess.run(training_op, feed_dict={training: True, X: X_batch, y: y_batch})
        acc_train = accuracy.eval(feed_dict={X: X_batch, y: y_batch})
        acc_val = accuracy.eval(feed_dict={X: mnist.validation.images, y: mnist.validation.labels})

        print(epoch, "Train accuracy: ", acc_train, "Val accuracy: ", acc_val)

    save_patch = saver.save(sess, "./cnn_fully_connected.ckpt")