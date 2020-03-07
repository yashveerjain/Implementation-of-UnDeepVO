import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D,Conv2DTranspose,UpSampling2D,MaxPooling2D,Concatenate,BatchNormalization,Flatten
from tensorflow.keras.layers import Dense,Layer
import numpy as np

"""
Well here also we are feeding the image both current and the next frame so its input is same and then we will use the output to 
compute the temporal loss and with the depth taken from the prediction from the depth estimation architecture 
The image in both will be the right image rather than the left as image as the it is stated in the undeepVO for temporal loss not the 
spatial loss as in spatial loss will take both left and right image pose output to compute it.

so the input shape for these architecture is 
(B,H,W,C*2)
"""
#previous thought design of pose estimation 
class convolution_pose(Model):
    def __init__(self,kernel,n_filter,pad,activate):
        super(convolution_pose,self).__init__()
        self.conv1 = Conv2D(n_filter,kernel,padding=pad,activation=activate)
        self.conv2 = Conv2D(n_filter,kernel,padding=pad,activation=activate)
        self.activation = tf.keras.layers.ReLU()
    def call(self,input_tensor):
        x = self.conv1(input_tensor)
        x = self.activation(x)
        x = self.conv2(x)
        x = self.activation(x)
        return x
class pose_est(Model):
    def __init__(self):
        super(pose_est,self).__init__()
        padding = 'valid'
        self.conv_layer1 = convolution_pose((7,7),16,pad=padding,activate='relu')
        self.conv_layer2 = convolution_pose((5,5),32,pad=padding,activate='relu')
        self.conv_layer3 = convolution_pose((3,3),64,pad=padding,activate='relu')
        self.conv_layer4 = convolution_pose((3,3),128,pad=padding,activate='relu')
        self.conv_layer5 = convolution_pose((3,3),256,pad=padding,activate='relu')
        self.conv_layer6 = convolution_pose((3,3),256,pad=padding,activate='relu')
        #self.conv_layer7_t = Conv2D(512,(3,3),padding='same')
        #self.conv_layer7_r = Conv2D(512,(3,3),padding='same')
        self.conv_layer7 = convolution_pose((3,3),512,pad=padding,activate='relu')
        #self.activate = tf.keras.layers.ReLU()
        
        self.flat = Flatten()
        self.denset1 = Dense(512,activation='relu')
        self.denset2 = Dense(512,activation='relu')
        self.denset3 = Dense(3)
        self.denser1 = Dense(512,activation='relu')
        self.denser2 = Dense(512,activation='relu')
        self.denser3 = Dense(3)
        
    def call(self,input_tensor):
        x = self.conv_layer1(input_tensor)
        x = self.conv_layer2(x)
        x = self.conv_layer3(x)
        x = self.conv_layer4(x)
        x = self.conv_layer5(x)
        x = self.conv_layer6(x)
        x = self.conv_layer7(x)
        x = self.flat(x)

        #t = self.conv_layer7_t(x)
        #t = self.activate(t)
        
        t = self.denset1(x)
        t = self.denset2(t)
        t = self.denset3(t)

        #r = self.conv_layer7_r(x)
        #r = self.activate(r)
        #r = self.flat(r)
        r = self.denser1(x)
        r = self.denser2(r)
        r = self.denser3(r)
        
        return x,r

