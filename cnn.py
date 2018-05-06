import tensorflow as tf

n_inputs = 28*28 #MNIST images are 28x28
n_filters_conv1 = 32
n_filters_conv2 = 64
n_outputs = 10
n_fc1 = 1024

X = tf.placeholder(tf.float32, shape=(None, n_inputs), name="X")
y = tf.placeholder(tf.int64, shape=(None), name="y")


with tf.name_scope("cnn"):
    X_reshaped = tf.reshape(X, shape=(-1, 28, 28, 1)) #reshape the image to 28x28x1 (only one channel)
    conv1 = tf.layers.conv2d(X_reshaped, filters=n_filters_conv1, kernel_size=[7,7], padding="SAME",
                            activation=tf.nn.relu, name="conv1")
    conv2 = tf.layers.conv2d(conv1, filters=n_filters_conv2, kernel_size=[3, 3], padding="SAME",
                            activation=tf.nn.relu, name="conv2")

with tf.name_scope("pool3"):
    max_pool = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1,2,2,1], padding="VALID")
    #max_pool with ksize=[1, 2, 2, 1] discard half pixels in height, and half in width, so the image is now 14x14
    max_pool_reshaped = tf.reshape(max_pool, shape=(-1,14 * 14 * n_filters_conv2))
    max_pool_flat = tf.layers.dense(max_pool_reshaped, n_outputs, name="outputs") #output before going through softmax activation

with tf.name_scope("fc1"):
    fc1 = tf.layers.dense(max_pool_flat, n_fc1, activation=tf.nn.relu, name="fc1")

with tf.name_scope("output"):
    logits = tf.layers.dense(fc1, n_outputs, name="output")
    Y_proba = tf.nn.softmax(logits, name="Y_proba")

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
            sess.run(training_op, feed_dict={X: X_batch, y: y_batch})
        acc_train = accuracy.eval(feed_dict={X: X_batch, y: y_batch})
        acc_val = accuracy.eval(feed_dict={X: mnist.validation.images, y: mnist.validation.labels})

        print(epoch, "Train accuracy: ", acc_train, "Val accuracy: ", acc_val)

    save_patch = saver.save(sess, "./cnn_fully_connected.ckpt")