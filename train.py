import numpy as np
import os
from glob import glob
from scipy import misc
import tensorflow as tf

from utils import *




def cropdata(images, factor=8):
    
    # Given np array of images, crop to dimensions usable by training net
    # i.e. dimensions must be divisible by power of 2

    factor = int(factor)
    if not bool(factor and not (factor&(factor-1))):
        print('factor is not power of 2')
        return False

    dimensions = images.shape
    height = dimensions[1]
    width = dimensions[2]
    targetH = int(height/factor)*factor
    targetW = int(width/factor)*factor
    delH = height - targetH
    delW = width - targetW
    y1 = int(delH*.5)
    y2 = y1+targetH
    x1 = int(delW*.5)
    x2 = x1+targetW
    if len(dimensions) is 4:
        cropped = images[:,y1:y2,x1:x2,:]
    else:
        cropped = images[:,y1:y2,x1:x2]

    return cropped



def downsample(images, factor=4):
    
    f = int(factor)
    dims = images.shape
    if len(dims) is 4:
        output = np.zeros([dims[0],dims[1]//f,dims[2]//f,dims[3]])
    else:
        output = np.zeros([dims[0],dims[1]//f,dims[2]//f])
        
    for i in range(dims[1]//f):
        for j in range(dims[2]//f):
            output[:,i,j] = np.mean( np.mean(
                images[:,(f*i):(f*i+f-1),(f*j):(f*j+f-1)],1),1)
            
    return output

    


    
""" Using functions from tf tutorial """

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




def upsample(x,height,width):
    return tf.image.resize_images(x,tf.cast([height,width],tf.int32))

def dropout(x,keep_prob):
    return tf.nn.dropout(x, keep_prob) 





def color_from_lines(x, imgH, imgW, doDrop=0):
    
    # Generate neural net from images x and return network output
    # Strategy adopted from Zhang https://arxiv.org/abs/1603.08511

    # Reshape images into tensor
    x_input = tf.reshape(x, [-1,imgH,imgW,1])

    # First convolution layer
    W_1 = weight_variable([5,5,1,32])
    b_1 = bias_variable([32])
    h_conv1 = tf.nn.relu( conv2d(x_input,W_1) + b_1)

    # Pooling (downsample by 2)
    # Maybe try averaging instead?
    h_pool1 = maxpool(h_conv1)

    # Second convolution layer
    W_2 = weight_variable([5,5,32,64])
    b_2 = bias_variable([64])
    h_conv2 = tf.nn.relu( conv2d(h_pool1,W_2) + b_2)

    # Second pooling
    h_pool2 = maxpool(h_conv2)

    # Third convolution layer
    W_3 = weight_variable([5,5,64,128])
    b_3 = bias_variable([128])
    h_conv3 = tf.nn.relu( conv2d(h_pool2,W_3) + b_3)
    
    # Fourth convolution layer
    W_4 = weight_variable([5,5,128,128])
    b_4 = bias_variable([128])
    h_conv4 = tf.nn.relu( conv2d(h_conv3,W_4) + b_4)

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
    h_conv5 = tf.nn.relu( conv2d(h_up4,W_5) + b_5)

    # Use residual info from previous images with this resolution
    reweight_5 = weight_variable([5,5,64,64])
    h_combine5 = h_conv5 + tf.nn.relu(conv2d(h_conv2,reweight_5))

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
    y_out = tf.nn.relu( conv2d(h_up5,W_6) + conv2d(h_conv1,reweight_6) + b_6)

    return y_out, keep_prob




def get_training_data():

    files = glob(os.path.join("training","colors","*.png"))
    colors = np.array([misc.imread(file,mode='RGB') for file in files])
    lines = colors[:,:,:,0]

    for i in range(colors.shape[0]):
        lines[i] = rgb2gray( np.array(
            misc.imread(os.path.join( "training","lines", os.path.split(files[i])[1] ),
                        mode='RGB') ))

    return lines, colors




def get_testing_data():

    files = glob(os.path.join("testing","colors","*.png"))
    colors = np.array([misc.imread(file,mode='RGB') for file in files])
    lines = colors[:,:,:,0]

    for i in range(colors.shape[0]):
        lines[i] = rgb2gray( np.array(
            misc.imread(os.path.join( "testing","lines", os.path.split(files[i])[1] ),
                        mode='RGB') ))

    return lines, colors




def image_closeness(imgset1,imgset2):

    images = tf.reduce_mean(tf.square(imgset1 - imgset2),[1,2,3])
    return tf.reduce_mean( tf.tanh(images) )



def image_match(imgset1,imgset2,window=10):

    pixel_max = tf.maximum(tf.abs(imgset1-imgset2),(imgset1*0.+window))
    pixel_flip = tf.exp( window - pixel_max )
    image_matches = tf.reduce_mean(pixel_flip,[1,2,3])
    return image_matches




def main(epochs=2000):

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

    y_out, keep_prob = color_from_lines(x, int(imgH), int(imgW), 1)

    objective = image_closeness(y_, y_out)
    train_step = tf.train.AdamOptimizer(1e-4).minimize(objective)
    correct_prediction = image_match(y_, y_out)
    accuracy = tf.reduce_mean(correct_prediction)

    print("Running session... ")
    batchsize = 15
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
            
            if i % 100 == 0:
                train_accuracy = accuracy.eval(feed_dict={
                    x: line_in, y_: color_in, keep_prob: 1.0})
                print('step %d, training accuracy %g' % (i, train_accuracy))
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
            os.path.join( "training","guesses","%05d.png" % j ), guesses[i])
        
    return guesses




    
