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
    left_generate_image = generate_image_left(imageright_k,right_disparity_k)
    right_generate_image = generate_image_right(imageleft_k,left_disparity_k)
    ##photometric_loss(left_image,right_generate_image,right_image,left_generate_image):
    lambd_s = 0.5 #weight is used for ssim and l1 loss
    loss_ssim_left = tf.reduce_mean(SSIM(imageleft_k,left_generate_image))
    loss_l1_left = (tf.reduce_mean(tf.abs(imageleft_k-left_generate_image)))
    loss_left = lambd_s*loss_ssim_left+(1-lambd_s)*loss_l1_left

    loss_ssim_right = tf.reduce_mean(SSIM(imageright_k,right_generate_image))   
    loss_l1_right = (tf.reduce_mean(tf.abs(imageright_k-right_generate_image)))
    loss_right = lambd_s*loss_ssim_right+(1-lambd_s)*loss_l1_right

    photo_loss = loss_left+loss_right
    
    ##Disparity_consistency_loss or #LR consistency loss
    left_to_right = generate_image_right(left_disparity_k,right_disparity_k)
    right_to_left = generate_image_left(right_disparity_k,left_disparity_k)

    loss_disp_left = tf.reduce_mean(tf.abs(left_to_right-right_disparity_k))
    loss_disp_right = tf.reduce_mean(tf.abs(right_to_left-left_disparity_k))

    dc_loss = loss_disp_left+loss_disp_right

    return dc_loss+photo_loss
 

def pose_est_loss(rot_right,rot_left,t_left,t_right,pk,pk1,intrinsics,depth):
    """
    argument :
    rot_right,t_right is the rotation and translation matrix when feeding the right image of the stereo each consist of shape [batch,3]
    rot_left,t_left is the rotation and translation matrix when feeding the left image of the stereo
    pk is the pixel in the current frame (of right image) [batch,image_height,image_width,3]
    pk1 which is p_k+1 is the next frame pixel or next image (of right image) [batch,image_height,image_width,3]
    intrinsic is the intrinsic parameter of the camera [batch,3,3]
    depth the depth is the prediction from the depth_estimation architecture [batch,image_height,image_width,2]
    """
    ##Spatial loss
    #Pose consistency loss
    lambd_p = 0.5
    lambd_o = 0.5

    photo_spatial_loss = lambd_p*tf.reduce_mean(tf.abs(t_right-t_left))+lambd_o*tf.reduce_mean(tf.abs(rot_right-rot_left))

    ##Temporal loss
    translate = t_right
    rotate = rot_right
    #loss for predicting the pk+1
    lambd = 0.5
    pk1_pred = projective_inverse_warp(pk, depth, translate,rotate, intrinsics)
    ssim_loss_pk1 = tf.reduce_mean(SSIM(pk1,pk1_pred))
    l1_loss_pk1 = tf.reduce_mean(tf.abs(tf.subtract(pk1_pred,pk1)))
    loss_pk1 = lambd*ssim_loss_pk1+(1-lambd)*l1_loss_pk1
    
    ##loss for predicting the pk
    lambd = 0.5
    pk_pred = projective_inverse_warp(pk1, depth, translate,rotate, intrinsics,inv=True)
    ssim_loss_pk = tf.reduce_mean(SSIM(pk,pk_pred))
    l1_loss_pk = tf.reduce_mean(tf.abs(tf.subtract(pk_pred,pk)))
    loss_pk = lambd*ssim_loss_pk+(1-lambd)*l1_loss_pk
    
    photo_temp_loss = loss_pk+loss_pk1
    
    #3D geometry loss
    ##TODO
    
    total_loss = photo_spatial_loss+photo_temp_loss
    return  total_loss
    
