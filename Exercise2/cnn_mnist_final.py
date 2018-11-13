from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import argparse
import gzip
import json
import os
import pickle
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


def one_hot(labels):
    """this creates a one hot encoding from a flat vector:
    i.e. given y = [0,2,1]
    it creates y_one_hot = [[1,0,0], [0,0,1], [0,1,0]]
    """
    classes = np.unique(labels)
    n_classes = classes.size
    one_hot_labels = np.zeros(labels.shape + (n_classes,))
    for c in classes:
        one_hot_labels[labels == c, c] = 1
    return one_hot_labels


def mnist(datasets_dir='./data'):
    if not os.path.exists(datasets_dir):
        os.mkdir(datasets_dir)
    data_file = os.path.join(datasets_dir, 'mnist.pkl.gz')

    if not os.path.exists(data_file):
        print('... downloading MNIST from the web')
        try:
            
            import urllib
            urllib.urlretrieve('http://google.com')
        except AttributeError:
            import urllib.request as urllib
        url = 'http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz'
        urllib.urlretrieve(url, data_file)

    print('... loading data')
    
    # Load the dataset
    
    f = gzip.open(data_file, 'rb')
    try:
        train_set, valid_set, test_set = pickle.load(f, encoding="latin1")
        
    except TypeError:
        train_set, valid_set, test_set = pickle.load(f)
    f.close()

    test_x, test_y = test_set
    test_x = test_x.astype('float32')
    test_x = test_x.astype('float32').reshape(test_x.shape[0], 28, 28, 1)
    test_y = test_y.astype('int32')
    valid_x, valid_y = valid_set
    #valid_x = valid_x.astype('float32')
    valid_x = valid_x.astype('float32').reshape(valid_x.shape[0], 28, 28, 1)
    valid_y = valid_y.astype('int32')
    train_x, train_y = train_set
    train_x = train_x.astype('float32').reshape(train_x.shape[0], 28, 28, 1)
    train_y = train_y.astype('int32')
    print('... done loading data')

   
    return train_x, one_hot(train_y), valid_x, one_hot(valid_y), test_x, one_hot(test_y)

    
    #lr = 0.01
    #epochs = 20
    #batch_size = 2


    
def train_and_validate(x_train, y_train, x_valid, y_valid, num_epochs, lr, num_filters, batch_size, filter_size):

    #creating placeholders for input and output
    x = tf.placeholder(tf.float32, shape = ([None,28, 28, 1]),name = 'x')
    y = tf.placeholder(tf.int32, shape = ([None,10]),name = 'y')
    

    #learned from tutorial
    #convolution layer1
    conv_one = tf.layers.conv2d(inputs=x,filters=num_filters,kernel_size=[3, 3],padding="same",activation=tf.nn.relu)
    print("conv one=",conv_one.shape)

    #pooling layer1
    pool_one = tf.layers.max_pooling2d(inputs=conv_one, pool_size=[2, 2], strides=1)
    print("pool one=",pool_one.shape)


    #convolution layer2
    conv_two = tf.layers.conv2d(inputs=pool_one,filters=num_filters,kernel_size=[3, 3],padding="same",activation=tf.nn.relu)
    print("conv two=",conv_two.shape)

    #pooling layer2
    pool_two = tf.layers.max_pooling2d(inputs=conv_two, pool_size=[2, 2], strides=1)
    print("pool two=",pool_two.shape)

    #flattening the pooling layer2 
    pool_two_flat = tf.reshape(pool_two, [-1, 26 * 26 * num_filters])
    print("pool_two_flat=",pool_two_flat.shape)
    
    #fully connected layer followed by relu activation function
    dense_layer = tf.layers.dense(inputs=pool_two_flat, units=128, activation=tf.nn.relu)
    print("dense_layer=",dense_layer.shape)

    #softmax layer
    softmax_layer = tf.layers.dense(inputs=dense_layer, units=10)
    print("softmax_layer=",softmax_layer.shape)
    
    #crossentropy loss function
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=softmax_layer, labels=y))
  
    #optimiser function
    optimiser = tf.train.GradientDescentOptimizer(learning_rate=lr).minimize(loss)

    #accuracy	
    correct_prediction = tf.equal(tf.argmax(softmax_layer, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    #saving the model
    model = tf.train.Saver()

    init_op = tf.global_variables_initializer()
    
    #calculating loss and accuracy for every epoch
    with tf.Session() as sess:
        
        # initialise the variables
        sess.run(init_op)
        
        #number of batches
        total_batch = int(len(y_train) / batch_size)
        print (total_batch)
        
        for epoch in range(num_epochs):
            avg_cost = 0
            avg_costv = 0
            for i in range(total_batch):
                
                #creating training batches
                x_batch = x_train[i*batch_size: (i+1)*batch_size]
                y_batch = y_train[i*batch_size: (i+1)*batch_size]
                _, cost = sess.run([optimiser, loss], feed_dict={x: x_batch, y: y_batch})
                avg_cost += cost / total_batch
                
            #Training accuracy
            train_acc = sess.run(accuracy, feed_dict={x: x_batch, y: y_batch})
            print("Epoch:", (epoch + 1), "train cost =", "{:.3f}".format(avg_cost), "train accuracy: {:.3f}".format(train_acc))
        
            #Validation accuracy
            valid_acc = sess.run(accuracy, feed_dict={x: x_validbatch, y: y_validbatch})
            learning_curve.append(valid_acc)
            print("Epoch:", (epoch + 1), "valid accuracy: {:.3f}".format(valid_acc))
        model.save(sess,'./my_cnn_model.ckpt')
        print("Validation done!")
        valid_error = 1 - np.array(learning_curve)
        print(valid_error)
        
        #return learning curve, 
        return learning_curve, accuracy

    
def test(x_test, y_test, model):

    sess = tf.Session()
    #importing saved model
    new_saver = tf.train.import_meta_graph('./my_cnn_model.ckpt.meta')

    #loading parameters
    
    new_saver.restore(sess, tf.train.latest_checkpoint('./'))
    graph = tf.get_default_graph()
    x = graph.get_tensor_by_name("x:0")
    y = graph.get_tensor_by_name("y:0")

    #feeding test data
    feed_dict ={x: x_test, y: y_test}

    #run saved model with new fed data
    test_acc = sess.run(model,feed_dict)
    test_err = 1 - np.array(test_acc)
    return test_error

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_path", default="./", type=str, nargs="?",
        help="Path where the results will be stored")
    parser.add_argument("--input_path", default="./", type=str, nargs="?",
        help="Path where the data is located. If the data is not available it will be downloaded first")
    parser.add_argument("--learning_rate", default=1e-3, type=float, nargs="?", help="Learning rate for SGD")
    parser.add_argument("--num_filters", default=16, type=int, nargs="?",
        help="The number of filters for each convolution layer")
    parser.add_argument("--batch_size", default=64, type=int, nargs="?", help="Batch size for SGD")
    parser.add_argument("--epochs", default=1, type=int, nargs="?",
        help="Determines how many epochs the network will be trained")
    parser.add_argument("--run_id", default=0, type=int, nargs="?",
        help="Helps to identify different runs of an experiments")
    parser.add_argument("--filter_size", default=3, type=int, nargs="?",
        help="Filter width and height")
    args = parser.parse_args()


    # hyperparameters

    lr = args.learning_rate
    num_filters = args.num_filters
    batch_size = args.batch_size
    epochs = args.epochs
    filter_size = args.filter_size

    # train and test convolutional neural network
    x_train, y_train, x_valid, y_valid, x_test, y_test = mnist(args.input_path)  

    learningcurve, model = train_and_validate(x_train, y_train, x_valid, y_valid, epochs, lr, num_filters, batch_size, filter_size)

    test_error = test(x_test, y_test, model)

    #Threw an error while running
    # save results in a dictionary and write them into a .json file
    results = dict()
    results["lr"] = lr
    results["num_filters"] = num_filters
    results["batch_size"] = batch_size
    results["filter_size"] = filter_size
    results["learning_curve"] = learning_curve
    results["test_error"] = test_error

    path = os.path.join(args.output_path, "results")
    os.makedirs(path, exist_ok=True)

    fname = os.path.join(path, "results_run_%d.json" % args.run_id)

    fh = open(fname, "w")
    json.dump(results, fh)
    fh.close()
