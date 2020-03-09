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
        if idx>=N:
            break
        left_imgk = tf.image.decode_png(tf.io.read_file(left_image_dir+left[idx]))
        left_imgk = tf.image.convert_image_dtype(left_imgk, tf.float32)
        l_k = tf.image.resize(left_imgk,  [image_height, image_width])
        left_imgk1 = tf.image.decode_png(tf.io.read_file(left_image_dir+left[idx+1]))
        left_imgk1 = tf.image.convert_image_dtype(left_imgk1, tf.float32)
        l_k1 = tf.image.resize(left_imgk1,  [image_height, image_width])
            
        right_imgk = tf.image.decode_png(tf.io.read_file(right_image_dir+left[idx]))
        right_imgk = tf.image.convert_image_dtype(right_imgk, tf.float32)
        r_k = tf.image.resize(right_imgk,  [image_height, image_width])
        right_imgk1 = tf.image.decode_png(tf.io.read_file(right_image_dir+left[idx+1]))
        right_imgk1 = tf.image.convert_image_dtype(right_imgk1, tf.float32)
        r_k1 = tf.image.resize(right_imgk1,  [image_height, image_width])
            
        train_left_image = tf.concat([l_k,l_k1],axis=2)
        train_right_image = tf.concat([r_k,r_k1],axis=2)
            
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
        
        for j in  range(batch):
            left_imgk = tf.image.decode_png(tf.io.read_file(left_image_dir+left[idx+j]))
            left_imgk = tf.image.convert_image_dtype(left_imgk, tf.float32)
            l_k = tf.image.resize(left_imgk,  [image_height, image_width])
            left_imgk1 = tf.image.decode_png(tf.io.read_file(left_image_dir+left[idx+1+j]))
            left_imgk1 = tf.image.convert_image_dtype(left_imgk1, tf.float32)
            l_k1 = tf.image.resize(left_imgk1,  [image_height, image_width])
            
            right_imgk = tf.image.decode_png(tf.io.read_file(right_image_dir+left[idx+j]))
            right_imgk = tf.image.convert_image_dtype(right_imgk, tf.float32)
            r_k = tf.image.resize(right_imgk,  [image_height, image_width])
            right_imgk1 = tf.image.decode_png(tf.io.read_file(right_image_dir+left[idx+1+j]))
            right_imgk1 = tf.image.convert_image_dtype(right_imgk1, tf.float32)
            r_k1 = tf.image.resize(right_imgk1,  [image_height, image_width])
            
            train_left_image = tf.concat([l_k,l_k1],axis=2)
            train_right_image = tf.concat([r_k,r_k1],axis=2)
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
