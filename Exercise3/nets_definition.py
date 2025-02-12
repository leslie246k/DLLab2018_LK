from __future__ import division
import os
import time
import math
import random
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.contrib.layers.python.layers import utils

import tensorflow.contrib as tc 

from layers_slim import *



def FCN_Seg(self, is_training=True):

    #Set training hyper-parameters
    
    self.is_training = is_training
    self.normalizer = tc.layers.batch_norm
    self.bn_params = {'is_training': self.is_training}


    print("input", self.tgt_image)

    with tf.variable_scope('First_conv'):
        conv1 = tc.layers.conv2d(self.tgt_image, 32, 3, 1, normalizer_fn=self.normalizer, normalizer_params=self.bn_params)

    print("Conv1 shape")
    print(conv1.get_shape())

    x = inverted_bottleneck(conv1, 1, 16, 0,self.normalizer, self.bn_params, 1)
    #print("Conv 1")
    #print(x.get_shape())

    #180x180x24
    x = inverted_bottleneck(x, 6, 24, 1,self.normalizer, self.bn_params, 2)
    x = inverted_bottleneck(x, 6, 24, 0,self.normalizer, self.bn_params, 3)

    print("Block One dim ")
    print(x)

    DB2_skip_connection = x    
    #90x90x32
    x = inverted_bottleneck(x, 6, 32, 1,self.normalizer, self.bn_params, 4)
    x = inverted_bottleneck(x, 6, 32, 0,self.normalizer, self.bn_params, 5)

    print("Block Two dim ")
    print(x)

    DB3_skip_connection = x
    #45x45x96
    x = inverted_bottleneck(x, 6, 64, 1,self.normalizer, self.bn_params, 6)
    x = inverted_bottleneck(x, 6, 64, 0,self.normalizer, self.bn_params, 7)
    x = inverted_bottleneck(x, 6, 64, 0,self.normalizer, self.bn_params, 8)
    x = inverted_bottleneck(x, 6, 64, 0,self.normalizer, self.bn_params, 9)
    x = inverted_bottleneck(x, 6, 96, 0,self.normalizer, self.bn_params, 10)
    x = inverted_bottleneck(x, 6, 96, 0,self.normalizer, self.bn_params, 11)
    x = inverted_bottleneck(x, 6, 96, 0,self.normalizer, self.bn_params, 12)

    print("Block Three dim ")
    print(x)

    DB4_skip_connection = x
    #23x23x160
    x = inverted_bottleneck(x, 6, 160, 1,self.normalizer, self.bn_params, 13)
    x = inverted_bottleneck(x, 6, 160, 0,self.normalizer, self.bn_params, 14)
    x = inverted_bottleneck(x, 6, 160, 0,self.normalizer, self.bn_params, 15)

    print("Block Four dim ")
    print(x)

    #23x23x320
    x = inverted_bottleneck(x, 6, 320, 0,self.normalizer, self.bn_params, 16)

    print("Block Four dim ")
    print(x)
    

    # Configuration 1 - single upsampling layer
    if self.configuration == 1:

        #input is features named 'x'
        
        #ToDo(1.1)
        current_up5 = TransitionUp_elu(x, 120,16,'current_up5')
        if current_up5.shape[1]> self.tgt_image.shape[1]:
            current_up5 = crop(current_up5,self.tgt_image)
        
        End_maps_decoder1 = slim.conv2d(current_up5, self.N_classes, [1, 1], scope='Final_decoder') #(batchsize, width, height, N_classes)

        Reshaped_map = tf.reshape(End_maps_decoder1, (-1, self.N_classes))

        print("End map size Decoder: ")
        print(Reshaped_map)


    # Configuration 2 - single upsampling layer plus skip connection
    
    if self.configuration == 2:

        #input is features named 'x'

        #ToDo(2.1)           
        F_up_21 = TransitionUp_elu(x,120,2,'F_up_21')
        F_up_21 = crop(F_up_21,DB4_skip_connection)
        output = Concat_layers(F_up_21,DB4_skip_connection)
        output_21 = Convolution(output,256,3,'output_21')
        
        #ToDo(2.2)
        current_up3 = TransitionUp_elu(output_21,120,8,'current_up3')
        if current_up3.shape[1] > self.tgt_image.shape[1]:
            current_up3 = crop(current_up3,self.tgt_image)
            
        End_maps_decoder1 = slim.conv2d(current_up3, self.N_classes, [1, 1], scope='Final_decoder') #(batchsize, width, height, N_classes)

        Reshaped_map = tf.reshape(End_maps_decoder1, (-1, self.N_classes))

        print("End map size Decoder: ")
        print(Reshaped_map)


    # Configuration 3 - Two upsampling layer plus skip connection
    if self.configuration == 3:

        #input is features named 'x'
        
        #ToDo(3.1)
        F_up_31 = TransitionUp_elu(x,120,2,'F_up_31')
        F_up_31 = crop(F_up_31,DB4_skip_connection)
        output = Concat_layers(F_up_31,DB4_skip_connection)
        output_31 = Convolution(output,256,3,'output_31')
        
        #ToDo(3.2)
        F_up_32 = TransitionUp_elu(output_31,120,2,'F_up_32')
        F_up_32 = crop(F_up_32,DB3_skip_connection)
        output_32 = Concat_layers(F_up_32,DB3_skip_connection)
        
        #ToDo(3.3)
        current_up4 = TransitionUp_elu(output_32,120,4,'current_up4')
        if current_up4.shape[1] > self.tgt_image.shape[1]:
            current_up4 = crop(current_up4,self.tgt_image)

        End_maps_decoder1 = slim.conv2d(current_up4, self.N_classes, [1, 1], scope='Final_decoder') #(batchsize, width, height, N_classes)

        Reshaped_map = tf.reshape(End_maps_decoder1, (-1, self.N_classes))

        print("End map size Decoder: ")
        print(Reshaped_map)


    #Full configuration
        
    if self.configuration == 4:

        ######################################### DECODER Full #############################################
        
        #ToDo(4.1)
        F_up_41 = TransitionUp_elu(x,120,2,'F_up_41')
        F_up_41 = crop(F_up_41,DB4_skip_connection)
        output = Concat_layers(F_up_41,DB4_skip_connection)
        output_41 = Convolution(output,256,3,'output_41')

        #ToDo(4.2)
        F_up_42 = TransitionUp_elu(output_41,160,2,'F_up_42')
        F_up_42 = crop(F_up_42,DB3_skip_connection)
        output_42 = Concat_layers(F_up_42,DB3_skip_connection)
        
        #ToDo(4.3)
        F_up_43 = TransitionUp_elu(output_42,96,2,'F_up_43')
        F_up_43 = crop(F_up_43,DB2_skip_connection)
        output_43 = Concat_layers(F_up_43,DB2_skip_connection)

        #ToDo(4.4)
        current_up5 = TransitionUp_elu(output_43,120,4,'current_up5')
        if current_up5.shape[1] > self.tgt_image.shape[1]:
            current_up5 = crop(current_up5,self.tgt_image)

        End_maps_decoder1 = slim.conv2d(current_up5, self.N_classes, [1, 1], scope='Final_decoder') #(batchsize, width, height, N_classes)

        Reshaped_map = tf.reshape(End_maps_decoder1, (-1, self.N_classes))

        print("End map size Decoder: ")
        print(Reshaped_map)


    return Reshaped_map
