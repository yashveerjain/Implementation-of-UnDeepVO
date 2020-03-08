import os
import tensorflow as tf
import cv2

def data_loader(left_image_dir,right_image_dir,image_width,image_height,number_of_data=None):
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
