import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D,Conv2DTranspose,UpSampling2D,MaxPooling2D,Concatenate,BatchNormalization,Flatten
from tensorflow.keras.layers import Dense,Layer
import numpy as np


"""
In depth Estimation we are basically feeding the images of set of 2 that means the current frame and the next frame 
which is require to predict the next and current disparity, that's what the undeepVO has done hence the shape of input will be
(B,H,W,C*2) 
where 
B is batch number
H is the height of the image
W is the width of the image
C is the channel (generally 3 RGB) and multiply it by 2 for 2 frames current and the next one

Note : Important point to note that if are feeding the left image than we are feeding its current and next frame that's it. 
we are Not feeding the left and right image simultaneously
"""

##creating the conv and deconv layer for the depth estimation

#currently working with the class function
class convolution(Layer):
    def __init__(self,kernel,n_filter,pad,name):
        super(convolution,self).__init__()
        self.conv = Conv2D(n_filter,kernel,padding=pad,name='conv'+name)
        self.bn = BatchNormalization(trainable=False,name='bn'+name)
        self.activate = tf.keras.layers.ReLU()
        self.mp = MaxPooling2D()

    def call(self,input_shape):
        x = self.conv(input_shape)
        x = self.bn(x)
        x = self.activate(x)
        x = self.mp(x)

        return x


def resize_like(inputs, ref):
    iH, iW = inputs.shape[1], inputs.shape[2]
    rH, rW = ref.shape[1], ref.shape[2]
    #tf.print(rH,rW)
    if iH == rH and iW == rW:
        return inputs
    return tf.image.resize(inputs, (rH, rW))#,method='nearest')

class deconvolution(Layer):
    def __init__(self,kernel,n_filter,pad,name):
        super(deconvolution,self).__init__()
        self.deconv = Conv2DTranspose(n_filter,kernel,strides=(2,2),padding=pad)
        self.conv = Conv2D(n_filter,kernel,padding=pad,name='deconv'+name)
        self.upsample = UpSampling2D(size=(2,2))
        self.concat = Concatenate()
        self.activate = tf.keras.layers.ReLU()

    def call(self,input_tensor,concat_tensor=None,deconv=False):
        if deconv :
            x = self.deconv(input_tensor)
            x = self.activate(x)
        else:
            x = self.upsample(input_tensor)
            #tf.print(x.shape)
            x = resize_like(x,concat_tensor)
            x = self.concat([x,concat_tensor])
            x = self.conv(x)
        return x
"""
This is just a try to predict the depth from the model but not working right now and we are computing the disparity 
def depth_layer(input_tensor):
imax_imin = tf.cast(tf.subtract(1,0),tf.float32)  #input max - input output
omax_omin = tf.cast(tf.subtract(800,100),tf.float32) #output max - output min
imin = tf.constant(0,dtype=tf.float32)
omin = tf.constant(100,dtype=tf.float32)
out = tf.add(tf.multiply(tf.divide((tf.subtract(input_tensor,0)),imax_imin),omax_omin),100)
#out = tf.clip_by_value(input_tensor,100,800)
return out
"""
class depth_estimation(Model):
    def __init__(self):
        super(depth_estimation,self).__init__()
        padding = 'same'
        self.Disp_const = 10 #it is the 'a' in the equation a*disp+epsilon assuming to be 10
        self.epsilon = 0.1 #not using currently-> ##1/800 where 800 is the max depth assuming B*f =1 

        self.conv1 = convolution((7,7),32,pad=padding,name='1')
        self.conv2 = convolution((5,5),64,pad=padding,name='2')
        self.conv3 = convolution((3,3),128,pad=padding,name='3')
        self.conv4 = convolution((3,3),256,pad=padding,name='4') 
        self.conv5 = convolution((3,3),512,pad=padding,name='5') 
        self.conv6 = convolution((3,3),512,pad=padding,name='6') 
        self.conv7 = convolution((3,3),512,pad=padding,name='7')

        self.deconv1 = deconvolution((3,3),512,pad=padding,name='1')
        self.deconv2 = deconvolution((3,3),512,pad=padding,name='2')
        self.deconv3 = deconvolution((3,3),256,pad=padding,name='3')
        self.deconv4 = deconvolution((3,3),128,pad=padding,name='4')
        self.deconv5 = deconvolution((3,3),64,pad=padding,name='5')
        self.deconv6 = deconvolution((3,3),32,pad=padding,name='6')
        self.deconv7 = deconvolution((3,3),16,pad=padding,name='7')
        self.disp = Conv2D(2,(3,3),activation = 'sigmoid',padding='same')

    def call(self,input_tensor):

        cx1 = self.conv1(input_tensor)
        cx2 = self.conv2(cx1)
        cx3 = self.conv3(cx2)
        cx4 = self.conv4(cx3)
        cx5 = self.conv5(cx4)
        cx6 = self.conv6(cx5)
        cx7 = self.conv7(cx6)

        dcx1 = self.deconv1(cx7,concat_tensor=cx6)
        dcx2 = self.deconv2(dcx1,cx5)
        dcx3 = self.deconv3(dcx2,cx4)
        dcx4 = self.deconv4(dcx3,cx3)
        dcx5 = self.deconv5(dcx4,cx2)
        dcx6 = self.deconv6(dcx5,cx1)
        dcx7 = self.deconv7(dcx6,deconv=True)
        #dcx1 = tf.keras.layers.Concatenate()([cx6,dcx1])
        disp = self.Disp_const*self.disp(dcx7)+self.epsilon
        #disp = None

        #dep = depth_layer(dep)
        return dcx7,disp
