import os
import tensorflow as tf
import cv2

def data_loader_without_batch(left_image_dir,right_image_dir,image_width,image_height,number_of_data=None):
    training_data = []
    left = sorted(os.listdir(left_image_dir))
    right = sorted(os.listdir(right_image_dir))
    if number_of_data == None:
        N = len(left)-2
    else:
        N=number_of_data
    for idx in range(len(left)):
        if idx==N:
            break
        left_imgk = cv2.imread(left_image_dir+left[idx])
        left_imgk = cv2.resize(left_imgk,(image_width,image_height))
        left_imgk1 = cv2.imread(left_image_dir+left[idx+1])
        left_imgk1 = cv2.resize(left_imgk1,(image_width,image_height))
        right_imgk = cv2.imread(right_image_dir+right[idx])
        right_imgk = cv2.resize(right_imgk,(image_width,image_height))
        right_imgk1 = cv2.imread(right_image_dir+right[idx+1])
        right_imgk1 = cv2.resize(right_imgk1,(image_width,image_height))
        l_k = tf.expand_dims(tf.constant(left_imgk,tf.float32),axis=0)
        l_k1 = tf.expand_dims(tf.constant(left_imgk1,tf.float32),axis=0)
        r_k = tf.expand_dims(tf.constant(right_imgk,tf.float32),axis=0)
        r_k1 = tf.expand_dims(tf.constant(right_imgk1,tf.float32),axis=0)
            
        train_left_image = tf.concat([l_k,l_k1],axis=-1)
        train_right_image = tf.concat([r_k,r_k1],axis=-1)
            
        training_data.append([train_left_image,train_right_image])
    return training_data

def data_loader_with_batch(left_image_dir,right_image_dir,image_width,image_height,batch=2,number_of_data=None):
    training_data = []
    left = sorted(os.listdir(left_image_dir))
    right = sorted(os.listdir(right_image_dir))
    if number_of_data == None:
        N = len(left)-2
    else:
        N=number_of_data
    idx=0
    while idx<N:
        #if idx>=N:
            #break
        for j in  range(batch):
            left_imgk = cv2.imread(left_image_dir+left[idx+j])
            left_imgk = cv2.resize(left_imgk,(image_width,image_height))
            left_imgk1 = cv2.imread(left_image_dir+left[idx+1+j])
            left_imgk1 = cv2.resize(left_imgk1,(image_width,image_height))
            right_imgk = cv2.imread(right_image_dir+right[idx+j])
            right_imgk = cv2.resize(right_imgk,(image_width,image_height))
            right_imgk1 = cv2.imread(right_image_dir+right[idx+1+j])
            right_imgk1 = cv2.resize(right_imgk1,(image_width,image_height))
            l_k = tf.constant(left_imgk,tf.float32)
            l_k1 = tf.constant(left_imgk1,tf.float32)
            r_k = tf.constant(right_imgk,tf.float32)
            r_k1 = tf.constant(right_imgk1,tf.float32)
            
            train_left_image = tf.concat([l_k,l_k1],axis=-1)
            train_right_image = tf.concat([r_k,r_k1],axis=-1)
            
            if j==0:
                train_batch_left = tf.expand_dims(train_left_image,axis=0)
                train_batch_right = tf.expand_dims(train_right_image,axis=0)
            else:
                train_left_image = tf.expand_dims(train_left_image,axis=0)
                train_right_image = tf.expand_dims(train_right_image,axis=0)
                train_batch_left = tf.concat([train_batch_left,train_left_image],axis=0)
                train_batch_right = tf.concat([train_batch_right,train_right_image],axis=0)
            idx+=1
            if idx>=N:
                break
        if idx<N:
            training_data.append([train_batch_left,train_batch_right])
    return training_data