from __future__ import print_function
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import tflearn
import numpy as np
import sys, os
import time

class TflearnLSTM():
    def __init__(self, h_size=128, n_inputs=28, n_steps=28, n_classes=10, l_r=0.001):
        # parameters init
        l_r = l_r
        self.n_inputs = n_inputs
        self.n_steps = n_steps
        n_classes = n_classes
        self.model_dir = 'model/tflearn/lstm'

        ## build graph
        tf.reset_default_graph()
        tflearn.init_graph(gpu_memory_fraction=0.1)
        X = tflearn.input_data(shape=[None, n_steps, n_inputs], name='input')
        lstm = tflearn.lstm(X, h_size, dynamic=True, name='lstm')
        dense = tflearn.fully_connected(lstm, n_classes, activation='softmax', name='dense')
        classifier = tflearn.regression(dense, optimizer='adam', loss='categorical_crossentropy', metric='R2', learning_rate=l_r)
        self.estimators = tflearn.DNN(classifier)

    def fit(self, X_data, Y_data, n_epoch=10, batch_size=128):
        self.estimators.fit(X_data, Y_data, n_epoch, show_metric=True, snapshot_epoch=False, batch_size=batch_size)
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        self.estimators.save('%s/model.ckpt' % self.model_dir)
        print("Model saved in file: %s" % self.model_dir)

    def predict(self, X_test):
        self.estimators.load('%s/model.ckpt' % self.model_dir)
        return self.estimators.predict(X_test)

def main():
    #load mnist data
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
    tfl_lstm = TflearnLSTM()
    t1 = time.time()
    tfl_lstm.fit(mnist.train.images.reshape(-1, 28, 28), mnist.train.labels)
    t2 = time.time()
    print('training time: %s' % (t2-t1))
    pred = tfl_lstm.predict(mnist.test.images.reshape(-1, 28, 28))
    t3 = time.time()
    print('predict time: %s' % (t3-t2))
    test_lab = mnist.test.labels
    print("accuracy: ", np.mean(np.equal(np.argmax(pred,1), np.argmax(test_lab,1)))*100)

if __name__ == '__main__':
    main()
