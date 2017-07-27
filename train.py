import numpy as np
import os
from glob import glob
from scipy import misc
import tensorflow as tf
import csv

from utils import *

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'





    
""" Functions borrowed from tf tutorial """

def conv2d(x,W):
    return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME')

def maxpool(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)



""" New functions """

def upsample(x,height,width):
    return tf.image.resize_images(x,tf.cast([height,width],tf.int32))

def dropout(x,keep_prob):
    return tf.nn.dropout(x, keep_prob)

def adjust(x):
    return tf.nn.relu(x)






""" CNN constructors """


def color_from_lines(x, imgH, imgW, doDrop=0):
    
    # Generate neural net from images x and return network output
    # Strategy adopted from Zhang https://arxiv.org/abs/1603.08511

    # Reshape images into tensor
    x_input = tf.reshape(x, [-1,imgH,imgW,1])

    # First convolution layer
    W_1 = weight_variable([5,5,1,32])
    b_1 = bias_variable([32])
    h_conv1 = adjust( conv2d(x_input,W_1) + b_1)

    # Pooling (downsample by 2)
    # Maybe try averaging instead?
    h_pool1 = maxpool(h_conv1)

    # Second convolution layer
    W_2 = weight_variable([5,5,32,64])
    b_2 = bias_variable([64])
    h_conv2 = adjust( conv2d(h_pool1,W_2) + b_2)

    # Second pooling
    h_pool2 = maxpool(h_conv2)

    # Third convolution layer
    W_3 = weight_variable([5,5,64,128])
    b_3 = bias_variable([128])
    h_conv3 = adjust( conv2d(h_pool2,W_3) + b_3)
    
    # Fourth convolution layer
    W_4 = weight_variable([5,5,128,128])
    b_4 = bias_variable([128])
    h_conv4 = adjust( conv2d(h_conv3,W_4) + b_4)

    # Upsample images
    keep_prob = tf.placeholder(tf.float32)
    if doDrop >= 2:
        h_drop4 = dropout(h_conv4, keep_prob)
        h_up4 = upsample(h_drop4,imgH/2,imgW/2)
    else:
        h_up4 = upsample(h_conv4,imgH/2,imgW/2)

    # Fifth convolution layer
    W_5 = weight_variable([5,5,128,64])
    b_5 = bias_variable([64])
    h_conv5 = adjust( conv2d(h_up4,W_5) + b_5)

    # Use residual info from previous images with this resolution
    reweight_5 = weight_variable([5,5,64,64])
    h_combine5 = h_conv5 + adjust(conv2d(h_conv2,reweight_5))

    # Second upsampling
    if doDrop >= 1:
        h_drop5 = dropout(h_combine5, keep_prob)
        h_up5 = upsample(h_drop5,imgH,imgW)
    else:
        h_up5 = upsample(h_combine5,imgH,imgW)


    # Map all weights to final 3 color channels
    W_6 = weight_variable([5,5,64,3])
    b_6 = bias_variable([3])
    reweight_6 = weight_variable([5,5,32,3])
    y_out = adjust( conv2d(h_up5,W_6) + conv2d(h_conv1,reweight_6) + b_6)

    return y_out, keep_prob




def labels_from_lines(x, imgH, imgW, doDrop=0, nclasses=7):
    
    # Generate neural net from images x and return network output
    # Strategy adopted from Zhang https://arxiv.org/abs/1603.08511

    # Reshape images into tensor
    x_input = tf.reshape(x, [-1,imgH,imgW,1])

    # First convolution layer
    W_1 = weight_variable([5,5,1,32])
    b_1 = bias_variable([32])
    h_conv1 = adjust( conv2d(x_input,W_1) + b_1)

    # Pooling (downsample by 2)
    # Maybe try averaging instead?
    h_pool1 = maxpool(h_conv1)

    # Second convolution layer
    W_2 = weight_variable([5,5,32,64])
    b_2 = bias_variable([64])
    h_conv2 = adjust( conv2d(h_pool1,W_2) + b_2)

    # Second pooling
    h_pool2 = maxpool(h_conv2)

    # Third convolution layer
    W_3 = weight_variable([5,5,64,128])
    b_3 = bias_variable([128])
    h_conv3 = adjust( conv2d(h_pool2,W_3) + b_3)
    
    # Fourth convolution layer
    W_4 = weight_variable([5,5,128,128])
    b_4 = bias_variable([128])
    h_conv4 = adjust( conv2d(h_conv3,W_4) + b_4)

    # Map feature maps to features
    W_f5 = weight_variable([imgH*imgW*8,1024])  # i.e. [imgH*imgW/16*128,1024]
    b_f5 = bias_variable([1024])
    h_flat4 = tf.reshape( h_conv4, [-1,imgH*imgW*8] )
    h_feat5 = tf.matmul(h_flat4, W_f5) + b_f5

    # Do or don't do dropout, and map features to classes
    W_f6 = weight_variable([1024,nclasses])
    b_f6 = bias_variable([nclasses])
    keep_prob = tf.placeholder(tf.float32)
    if doDrop >= 1:
        h_drop5 = dropout(h_feat5, keep_prob)
        y_out = tf.matmul(h_drop5, W_f6) + b_f6
    else:
        y_out = tf.matmul(h_feat5, W_f6) + b_f6

    return y_out, keep_prob







""" Utils for importing data """


def get_training_data():

    files = glob(os.path.join("training","colors","*.png"))
    colors = np.array([misc.imread(file,mode='RGB') for file in files])
    lines = colors[:,:,:,0]

    for i in range(colors.shape[0]):
        lines[i] = rgb2gray( np.array(
            misc.imread(os.path.join( "training","lines", os.path.split(files[i])[1] ),
                        mode='RGB') ))

    return lines/255., colors/255.


def get_testing_data():

    files = glob(os.path.join("testing","colors","*.png"))
    colors = np.array([misc.imread(file,mode='RGB') for file in files])
    lines = colors[:,:,:,0]

    for i in range(colors.shape[0]):
        lines[i] = rgb2gray( np.array(
            misc.imread(os.path.join( "testing","lines", os.path.split(files[i])[1] ),
                        mode='RGB') ))

    return lines/255., colors/255.


def get_labels(test=0):

    if test is 1:
        folder = "testing"
    else:
        folder = "training"
        
    with open(os.path.join(folder,"labels","temp.csv"), "r") as f:
        reader = csv.reader(f)
        labels = list(reader)

    i = 0
    while not ((labels[i][0] == '0000') or (labels[i][0] == '00000')):
        i += 1
    labels = labels[i:]

    vector = np.zeros( [len(labels),len(labels[0][1])], 'f')
    for i in range(len(labels)):
        vector[i,:] = np.fromstring(labels[i][1],'u1') - ord('0')

    return vector






""" Utils for saving/loading models """

def save_model(sess, saver, checkpoint):

    print("Saving model...")

    name = "labels_from_lines"
    if (checkpoint == 0):
        fname = name + ".ckpt"
        path = os.path.join("models",fname)
        savepath = saver.save(sess,path)
    else:
        path = os.path.join("models",name)
        savepath = saver.save(sess, path, global_step=checkpoint)

    print("Model saved in %s" % savepath)
              
    return True



def restore_model(sess, saver, checkpoint):

    name = "labels_from_lines"
    if (checkpoint == 0):
        fname = name + ".ckpt"
    else:
        fname = name + "-" + str(checkpoint)

    path = os.path.join("models",fname)
    if not os.path.isfile(path + ".index"):
        print("Model not found. Using default initialization, if any.")
    else:
        print("Restoring model...")
        saver.restore(sess, path)
        print("Model restored.")
              
    return True






""" Utils for constructing Objective functions """


def variance(img):

    return tf.reduce_mean( tf.square(img)) - tf.square(tf.reduce_mean(img))


def dot_product(y1,y2):

    return tf.reduce_sum(y1*y2) \
           / tf.sqrt( tf.reduce_sum(tf.square(y1)) * tf.reduce_sum(tf.square(y2)) )


def thresh_norm(x, threshold=.7):

    [_,thresh] = tf.meshgrid(tf.reduce_mean(x,[0]), tf.reduce_max(x,[1])*threshold)
    return x / thresh


def threshold_match(y1,y2, threshold=.7):

    [_,thresh1] = tf.meshgrid(tf.reduce_mean(y1,[0]), tf.reduce_max(y1,[1])*threshold)
    [_,thresh2] = tf.meshgrid(tf.reduce_mean(y2,[0]), tf.reduce_max(y2,[1])*threshold)

    return tf.reduce_mean( tf.cast( tf.equal( 
        tf.cast( tf.greater(y1, thresh1), tf.uint8),
        tf.cast( tf.greater(y2, thresh2), tf.uint8)\
                                            ), tf.float32), [1])


def image_closeness(imgset1,imgset2):

    colors = tf.reduce_mean(tf.square(imgset1-imgset2),[3])
    images = tf.reduce_mean(tf.exp(2.*colors),[1,2])
    #return tf.reduce_mean( tf.log(images+.1) )
    return tf.reduce_mean( images )


def image_match(imgset1,imgset2,window=0.05):

    pixel_max = tf.maximum(tf.abs(imgset1-imgset2),(imgset1*0.+window))
    pixel_flip = tf.exp( 10.*(window - pixel_max) )
    image_matches = tf.reduce_mean(pixel_flip,[1,2,3])
    return image_matches






""" Training routines """


def train_labels(epochs=2000, batchsize=20, update_int=100, load=-1, save=0,
                 drop=1, optimizer=1, return_all=False, **opt_params):

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

    print("Loading training data... ")
    [lines, _] = get_training_data()
    lines = downsample(cropdata(lines,32),8)
    labels = get_labels()
    [numimgs, imgH, imgW] = lines.shape
    nlabels = labels.shape[1]

    print("Loading testing data... ")
    [lines_test, _] = get_testing_data()
    lines_test = downsample(cropdata(lines_test,32),8)
    labels_test = get_labels(1)
    [num_test,_,_] = lines_test.shape

    print("Constructing graph... ")
    x = tf.placeholder(tf.float32, [None,imgH,imgW])
    y_ = tf.placeholder(tf.float32, [None,nlabels])

    y_out, keep_prob = labels_from_lines(x, int(imgH), int(imgW), drop, nlabels)
    
    y_norm = thresh_norm(y_out)
    y_int = tf.cast( y_norm > 1., tf.uint8)

    objective = tf.acos(dot_product(y_,y_out))
    if optimizer is 1:
        train_step = tf.train.AdamOptimizer(**opt_params).minimize(objective)
    elif optimizer is 2:
        train_step = tf.train.AdagradDAOptimizer(**opt_params).minimize(objective)
    else:
        train_step = tf.train.AdadeltaOptimizer(**opt_params).minimize(objective)
    correct_prediction = threshold_match(y_out, y_, .7)
    accuracy = tf.reduce_mean(correct_prediction)

    print("Running session... ")
    batchsize = 20
    if numimgs % batchsize == 0:
        offset = 3
    elif numimgs % batchsize == 5:
        offset = 4
    else:
        offset = 0

    savesize = 50
    writesize = 150
    flagout = np.zeros([savesize,nlabels+1])
    flagout[:,0] = np.arange(savesize)

    saver = tf.train.Saver()
    
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
       
        for i in range(epochs):
            index = (i*batchsize) % (numimgs - batchsize - offset)
            line_in = lines[index:(index+batchsize)]
            label_in = labels[index:(index+batchsize)]
            
            if i % update_int == 0:
                train_accuracy = accuracy.eval(feed_dict={
                    x: line_in, y_: label_in, keep_prob: 1.0})
                obj_val = objective.eval(feed_dict={
                    x: line_in, y_: label_in, keep_prob: 1.0})
                print('step %d, training set accuracy %8g, objective %8g' \
                      % (i, train_accuracy, obj_val))
                guesses_test = y_out.eval(feed_dict={
                    x:lines_test[0:savesize], y_:labels_test[0:savesize], keep_prob:1.0})
                np.savetxt(
                    os.path.join(
                        "training","guesses","test%05d.csv" % (i)), guesses_test, delimiter=",")

                # PROBABLY BETTER TO SAVE TRAINING DATA HERE - SWAP OUT?
                
                flags_test = y_int.eval(feed_dict={
                    x:lines_test[0:savesize], y_:labels_test[0:savesize], keep_prob:1.0})
                flagout[:,1:] = flags_test
                np.savetxt(
                    os.path.join(
                        "training","guesses","t_int%05d.csv" % (i)),
                        flagout, fmt='%04u,%u%u%u%u%u%u%u', delimiter=",")

            if (i == 1) and (load >= 0):
                restore_model(sess,saver,load)
                    
            train_step.run(feed_dict={x: line_in, y_: label_in, keep_prob: 0.5})

        print('test accuracy %g' % accuracy.eval(feed_dict={
            x: lines_test, y_: labels_test, keep_prob: 1.0}))

        if (save >= 0):
            save_model(sess,saver,save)
        
        guesses = y_norm.eval(feed_dict={
            x:lines[0:writesize], y_:labels[0:writesize], keep_prob:1.0})
        flags = y_int.eval(feed_dict={
            x:lines[0:writesize], y_:labels[0:writesize], keep_prob:1.0})

    np.savetxt( os.path.join("training","guesses","final.csv"), guesses, delimiter=",")
    flagout = np.zeros([writesize,nlabels+1])
    flagout[:,0] = np.arange(writesize)
    flagout[:,1:] = flags
    np.savetxt( os.path.join("training","guesses","final_int.csv"),
                flagout, fmt='%04u,%u%u%u%u%u%u%u',delimiter=",")

    if return_all:
        return guesses, labels[0:writesize], lines[0:writesize], \
               guesses_test, flags_test, lines_test[0:savesize]
    
    return guesses, labels[0:writesize], lines[0:writesize]




def main(epochs=2000, batchsize=20, update_int=100, drop=1, optimizer=1, **opt_params):

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

    print("Loading training data... ")
    [lines, colors] = get_training_data()
    lines = downsample(cropdata(lines,16))
    colors = downsample(cropdata(colors,16))
    [numimgs, imgH, imgW, _] = colors.shape

    print("Loading testing data... ")
    [lines_test, colors_test] = get_testing_data()
    lines_test = downsample(cropdata(lines_test,16))
    colors_test = downsample(cropdata(colors_test,16))
    [num_test,_,_] = lines_test.shape

    print("Constructing graph... ")
    x = tf.placeholder(tf.float32, [None,imgH,imgW])
    y_ = tf.placeholder(tf.float32, [None,imgH,imgW,3])

    y_out, keep_prob = color_from_lines(x, int(imgH), int(imgW), drop)

    objective = image_closeness(y_, y_out) \
                - 100.*variance(y_out) \
                - 1000.*tf.reduce_mean(y_out)
    if optimizer is 1:
        train_step = tf.train.AdamOptimizer(**opt_params).minimize(objective)
    elif optimizer is 2:
        train_step = tf.train.AdagradDAOptimizer(**opt_params).minimize(objective)
    else:
        train_step = tf.train.AdadeltaOptimizer(**opt_params).minimize(objective)
    correct_prediction = image_match(y_, y_out)
    accuracy = tf.reduce_mean(correct_prediction)

    print("Running session... ")
    batchsize = 20
    if numimgs % batchsize == 0:
        offset = 3
    elif numimgs % batchsize == 5:
        offset = 4
    else:
        offset = 0

    savesize = 15
    writesize = 50
        
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        
        for i in range(epochs):
            index = (i*batchsize) % (numimgs - batchsize - offset)
            line_in = lines[index:(index+batchsize)]
            color_in = colors[index:(index+batchsize)]
            
            if i % update_int == 0:
                train_accuracy = accuracy.eval(feed_dict={
                    x: line_in, y_: color_in, keep_prob: 1.0})
                obj_val = objective.eval(feed_dict={
                    x: line_in, y_: color_in, keep_prob: 1.0})
                print('step %d, accuracy %8g, objective %8g' \
                      % (i, train_accuracy, obj_val))
                guesses = y_out.eval(feed_dict={
                    x:lines_test[0:savesize], y_:colors_test[0:savesize], keep_prob:1.0})
                for j in range(savesize):
                    misc.imsave(
                        os.path.join(
                            "training","guesses","test%05d-%05d.png" % (j,i) ), guesses[j])
                    
            train_step.run(feed_dict={x: line_in, y_: color_in, keep_prob: 0.5})

        print('test accuracy %g' % accuracy.eval(feed_dict={
            x: lines_test, y_: colors_test, keep_prob: 1.0}))

        guesses = y_out.eval(feed_dict={
            x:lines[0:writesize], y_:colors[0:writesize], keep_prob:1.0})

    for j in range(writesize):
        misc.imsave(
            os.path.join( "training","guesses","%05d.png" % j ), guesses[j])
        
    return guesses


    
