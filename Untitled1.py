
# coding: utf-8

# In[1]:


import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import time


# In[2]:


with open("DSL-StrongPasswordData.csv") as f:
    data = np.array([line.split(",") for line in f.read().strip().split("\n")[1:]])
    
# Formatando os dados
np.random.seed(1234)
np.random.shuffle(data)
recordings, keystrokes = data[:,:3], data[:,3:].astype(float)
print(recordings[:2], "\n", keystrokes[:2])


# In[3]:


print(keystrokes.shape)


# In[4]:


users = sorted(list(set(recordings[:,0])))
print("Identificadores de usuários: {}".format(users))
print("Quantidade de usuários únicos: {}".format(len(users)))


# In[5]:


labels = np.array([users.index(rec) for rec in recordings[:,0]])


# In[6]:


labels


# In[7]:


split_data_idx = 20000
train_data = keystrokes[:split_data_idx]
test_data = keystrokes[split_data_idx:]

train_labels = labels[:split_data_idx]
test_labels = labels[split_data_idx:]
test_labels_one_hot = np.zeros((len(test_labels),51))
for j, idx in enumerate(test_labels):
    test_labels_one_hot[j][idx] = 1


# In[8]:


def next_batch(size, i):
    l = train_labels[i*size:(i+1)*size]
    onehot = np.zeros((len(l),51))
    for j, idx in enumerate(l):
        onehot[j][idx] = 1
    k = train_data[i*size:(i+1)*size]
    return k,onehot.astype(float)


# In[9]:


with tf.device("/device:GPU:0"):
    x = tf.placeholder(tf.float32, shape=[None, 31])
    y = tf.placeholder(tf.float32, shape=[None, 51])

    W = tf.get_variable("W", shape=[31, 51], initializer=tf.contrib.layers.xavier_initializer())
    b = tf.Variable(tf.zeros([51]), name="b")

    y_ = tf.matmul(x,W) + b

    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=y_))

    train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)


# In[10]:


def test_accuracy():
    with tf.device("/device:GPU:0"): 
        correct_prediction = tf.equal(tf.argmax(y_,1), tf.argmax(y,1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        return accuracy.eval(feed_dict={x: test_data, y: test_labels_one_hot})


# In[ ]:

# Add ops to save and restore all the variables.
saver = tf.train.Saver()

train_acc = []
test_acc = []
max_acc = 0
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for epoch in range(100):
        for i in range(200):
            batch = next_batch(100, i)
    #         print(batch[0].shape, batch[1].shape)
            train_step.run(feed_dict={x: batch[0], y: batch[1]})
            test_acc.append(test_accuracy())

            if test_acc[-1] > max_acc:
                max_acc = test_acc[-1]
                save_path = saver.save(sess, "weights/"+str(epoch)+" - "+str(time.time())+".ckpt")
                print("Model saved in file: %s" % save_path)

        print("Ep: {} - Acc: {}%".format(epoch, 100*test_acc[-1]))

        acc_f_name = "accuracy/"+str(epoch)+" - "+str(time.time())+".txt"
        with open(acc_f_name, mode="w") as f:
            f.write("\n".join(str(test_acc)))
            print("Model acc saved in file: %s" % acc_f_name)

