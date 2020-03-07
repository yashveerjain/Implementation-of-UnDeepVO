##losses for UnDeepVo
import tensorflow as tf
import numpy as np
from utils import *

##depth estimation loss spatial 
def dept_est_loss(depth_prediction_left,depth_prediction_right,left_image,right_image):
    B = 10 #base length (cm)
    f = .04 #focal length (cm)
    
    #there is the 4 disparity maps 2 for each stero image (i.e right stereo image and left stereo image)
    ##k and k+1 disparities for left image
    ##left image generates left disparity 
    left_disparity_k = depth_prediction_left[:,:,:,0] #(B*f)/depth_prediction_left[:,:,:,0] currently assuming the depth_prediction as disparity
    #tf.print(left_disparity_k.shape)
    left_disp_batch_num  = depth_prediction_left.shape[0]
    left_disp_height = depth_prediction_left.shape[1]
    left_disp_width = depth_prediction_left.shape[2]
    left_disparity_k = tf.reshape(left_disparity_k,[left_disp_batch_num,left_disp_height,left_disp_width,1])
    #print(left_disparity_k.shape)
    
    #left_disparity_k1 = (B*f)/depth_prediction_left[:,:,:,1]

    ##k and k+1 disparities for right image
    ##generate the right image from the right disparity
    #right_disparity_k1 = (B*f)/depth_prediction_right[:,:,:,1]
    right_disparity_k = depth_prediction_right[:,:,:,0]#(B*f)/depth_prediction_right[:,:,:,0]
    right_disp_batch_num  = depth_prediction_right.shape[0]
    right_disp_height = depth_prediction_right.shape[1]
    right_disp_width = depth_prediction_right.shape[2]
    right_disparity_k = tf.reshape(right_disparity_k,[right_disp_batch_num,right_disp_height,right_disp_width,1])
    #print(right_disparity_k.shape)
    #tf.print(right_disparity_k)
    
    ##take the current frame from the image means rather than taking all the channels just take the first 3 channel
    imageleft_k = tf.slice(left_image,[0,0,0,0],[-1,-1,-1,3]) #image right k
    imageright_k = tf.slice(right_image,[0,0,0,0],[-1,-1,-1,3]) #image right k

    imageleft_k1 = tf.slice(left_image,[0,0,0,3],[-1,-1,-1,-1])#image left k+1
    imageright_k1 = tf.slice(right_image,[0,0,0,3],[-1,-1,-1,-1])#image right k+1


    ##spatial loss
    left_generate_image = generate_image_left(right_image,right_disparity_k)
    right_generate_image = generate_image_right(left_image,left_disparity_k)
    ##photometric_loss(left_image,right_generate_image,right_image,left_generate_image):
    lambd_s = 0.5 #weight is used for ssim and l1 loss
    loss_ssim_left = tf.reduce_mean(SSIM(left_image,left_generate_image))
    loss_l1_left = (tf.reduce_mean(tf.abs(left_image-left_generate_image)))
    loss_left = lambd_s*loss_ssim_left+(1-lambd_s)*loss_l1_left

    loss_ssim_right = tf.reduce_mean(SSIM(right_image,right_generate_image))   
    loss_l1_right = (tf.reduce_mean(tf.abs(right_image-right_generate_image)))
    loss_right = lambd_s*loss_ssim_right+(1-lambd_s)*loss_l1_right

    photo_loss = loss_left+loss_right
    
    ##Disparity_consistency_loss or #LR consistency loss
    left_to_right = generate_image_right(left_disparity_k,right_disparity_k)
    right_to_left = generate_image_left(right_disparity_k,left_disparity_k)

    loss_disp_left = tf.reduce_mean(tf.abs(left_to_right-right_disparity_k))
    loss_disp_right = tf.reduce_mean(tf.abs(right_to_left-left_disparity_k))

    dc_loss = loss_disp_left+loss_disp_right

    return dc_loss+photo_loss
 
