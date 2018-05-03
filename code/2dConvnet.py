#TODO: gets oom error, fix using batches?

import tensorflow as tf
import numpy as np
import os

NUMBER_OF_PATIENTS = 15
IMG_RESIZE = 50 #Pixel dimensions of axial slices after resize
NR_SLICES = 20 #NR of axial slices used for one volume

n_classes = 7

x = tf.placeholder('float')
y = tf.placeholder('float')

keep_rate = 0.8
keep_prob = tf.placeholder(tf.float32)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME')

def maxpool2d(x):
    #                        size of window         movement of window
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')



def convolutional_neural_network(x):
    weights = {'W_conv1':tf.Variable(tf.random_normal([3,3,1,32])),
               'W_conv2':tf.Variable(tf.random_normal([3,3,32,64])),
               'W_fc':tf.Variable(tf.random_normal([54080,1024])),
               'out':tf.Variable(tf.random_normal([1024, n_classes]))}

    biases = {'b_conv1':tf.Variable(tf.random_normal([32])),
               'b_conv2':tf.Variable(tf.random_normal([64])),
               'b_fc':tf.Variable(tf.random_normal([1024])),
               'out':tf.Variable(tf.random_normal([n_classes]))}

    x = tf.reshape(x, shape=[-1, IMG_RESIZE, IMG_RESIZE, 1])

    conv1 = tf.nn.relu(conv2d(x, weights['W_conv1']) + biases['b_conv1'])
    conv1 = maxpool2d(conv1)
    
    conv2 = tf.nn.relu(conv2d(conv1, weights['W_conv2']) + biases['b_conv2'])
    conv2 = maxpool2d(conv2)

    fc = tf.reshape(conv2,[-1, 54080]) 
    fc = tf.nn.relu(tf.matmul(fc, weights['W_fc'])+biases['b_fc'])
    fc = tf.nn.dropout(fc, keep_rate)

    output = tf.matmul(fc, weights['out'])+biases['out']

    return output

def train_neural_network(x):
    for patient in range(0, NUMBER_OF_PATIENTS):
        dataset = np.load(os.path.join('/media/ruben/Seagate Expansion Drive/bachelorProject/code/data/2dData', '2dData-PAT'+str(patient)+'-50-50.npy'))
        train_data = dataset[:-5]
        validation_data = dataset[-5:]
        print(np.shape(dataset))
        
        '''
        prediction = convolutional_neural_network(x)
        cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits_v2(logits=prediction,labels=y))
        optimizer = tf.train.AdamOptimizer().minimize(cost)
        
        hm_epochs = 10
        with tf.Session() as sess:
            #sess.run(tf.initialize_all_variables()) #deprecated
            sess.run(tf.global_variables_initializer())

            for epoch in range(hm_epochs):
                epoch_loss = 0
                #data is too big, segment in smaller sets during reading
                for data in train_data:
                    X = data[0]
                    Y = data[1]
                    _, c = sess.run([optimizer, cost], feed_dict={x: X, y: Y})
                    epoch_loss += c

                print('Epoch', epoch, 'completed out of',hm_epochs,'loss:',epoch_loss)

            correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))

            accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
            print('Accuracy:',accuracy.eval({x:[i[0] for i in validation_data], y:[i[1] for i in validation_data]}))
        '''
train_neural_network(x)