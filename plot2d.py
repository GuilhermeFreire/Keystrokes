import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import time

with open("DSL-StrongPasswordData.csv") as f:
    data = np.array([line.split(",") for line in f.read().strip().split("\n")[1:]])


np.random.seed(1234)
np.random.shuffle(data)
recordings, keystrokes = data[:,:3], data[:,3:].astype(float)
print(recordings[:2], "\n", keystrokes[:2])


split_data_idx = 20000
train_data = keystrokes[:split_data_idx]
test_data = keystrokes[split_data_idx:]


def next_batch(size, i):
    return train_data[i*size:(i+1)*size]


def test_distance_f():
    with tf.device("/device:CPU:0"):
        sqrt_instruction = tf.sqrt(loss)
        return sqrt_instruction.eval(feed_dict={x: test_data})


with tf.device("/device:CPU:0"):
    x = tf.placeholder(tf.float32, shape=[None, 31])

    W_compress = tf.get_variable("W_compress", shape=[31, 2], initializer=tf.contrib.layers.xavier_initializer())
    b_compress = tf.Variable(tf.zeros([2]), name="b_compress")
    W_expand = tf.get_variable("W_expand", shape=[2, 31], initializer=tf.contrib.layers.xavier_initializer())
    b_expand = tf.Variable(tf.zeros([31]), name="b_expand")

    output = tf.add(tf.matmul(x, W_compress), b_compress)
    encoded = tf.nn.relu(output)
    decoded = tf.add(tf.matmul(encoded, W_expand), b_expand)

    loss = tf.losses.mean_squared_error(x, decoded)
    train = tf.train.AdamOptimizer().minimize(loss)


# Add ops to save and restore all the variables.
saver = tf.train.Saver()

test_distance = []
min_distance = float("inf")
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for epoch in range(1000000):
        for i in range(200):
            batch = next_batch(100, i)
            train.run(feed_dict={x: batch})
            test_distance.append(test_distance_f())

            if test_distance[-1] < min_distance:
                min_distance = test_distance[-1]
                save_path = saver.save(sess, "weights-plot2d/"+str(epoch)+" - "+str(time.time())+".ckpt")
                print("Model saved in file: %s" % save_path)

        print("Ep: {} - Distance: {}".format(epoch, test_distance[-1]))

        distance_f_name = "distance-plot2d/"+str(epoch)+" - "+str(time.time())+".txt"
        with open(distance_f_name, mode="w") as f:
            f.write("\n".join(str(test_distance)))
            print("Model Distance saved in file: %s" % distance_f_name)

