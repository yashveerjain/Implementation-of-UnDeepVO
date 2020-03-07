##losses for UnDeepVo
import tensorflow as tf
import numpy as np

def bilinear_sampler_1d_h(input_images, x_offset, wrap_mode='border', name='bilinear_sampler', **kwargs):
    def _repeat(x, n_repeats):
        #with tf.variable_scope('_repeat'):
        rep = tf.tile(tf.expand_dims(x, 1), [1, n_repeats])
        return tf.reshape(rep, [-1])

    def _interpolate(im, x, y):

        #with tf.variable_scope('_interpolate'):

      # handle both texture border types
        _edge_size = 0
        if _wrap_mode == 'border':
            _edge_size = 1
            im = tf.pad(im, [[0, 0], [1, 1], [1, 1], [0, 0]], mode='CONSTANT')
            x = x + _edge_size
            y = y + _edge_size
        elif _wrap_mode == 'edge':
            _edge_size = 0
        else:
            return None

        x = tf.clip_by_value(x, 0.0,  _width_f - 1 + 2 * _edge_size)

        x0_f = tf.floor(x)
        y0_f = tf.floor(y)
        x1_f = x0_f + 1

        x0 = tf.cast(x0_f, tf.int32)
        y0 = tf.cast(y0_f, tf.int32)
        x1 = tf.cast(tf.minimum(x1_f,  _width_f - 1 + 2 * _edge_size), tf.int32)

        dim2 = (_width + 2 * _edge_size)
        dim1 = (_width + 2 * _edge_size) * (_height + 2 * _edge_size)
        base = _repeat(tf.range(_num_batch) * dim1, _height * _width)
        base_y0 = base + y0 * dim2
        idx_l = base_y0 + x0
        idx_r = base_y0 + x1

        im_flat = tf.reshape(im, tf.stack([-1, _num_channels]))

        pix_l = tf.gather(im_flat, idx_l)
        pix_r = tf.gather(im_flat, idx_r)

        weight_l = tf.expand_dims(x1_f - x, 1)
        weight_r = tf.expand_dims(x - x0_f, 1)

        return weight_l * pix_l + weight_r * pix_r

    def _transform(input_images, x_offset):
        # grid of (x_t, y_t, 1), eq (1) in ref [1]
        x_t, y_t = tf.meshgrid(tf.linspace(0.0,   _width_f - 1.0,  _width),
                                tf.linspace(0.0 , _height_f - 1.0 , _height))

        x_t_flat = tf.reshape(x_t, (1, -1))
        y_t_flat = tf.reshape(y_t, (1, -1))

        x_t_flat = tf.tile(x_t_flat, tf.stack([_num_batch, 1]))
        y_t_flat = tf.tile(y_t_flat, tf.stack([_num_batch, 1]))

        x_t_flat = tf.reshape(x_t_flat, [-1])
        y_t_flat = tf.reshape(y_t_flat, [-1])

        x_t_flat = x_t_flat + tf.reshape(x_offset, [-1]) * _width_f

        input_transformed = _interpolate(input_images, x_t_flat, y_t_flat)

        output = tf.reshape(
            input_transformed, tf.stack([_num_batch, _height, _width, _num_channels]))
        return output


    _num_batch    = tf.shape(input_images)[0]
    _height       = tf.shape(input_images)[1]
    _width        = tf.shape(input_images)[2]
    _num_channels = tf.shape(input_images)[3]

    _height_f = tf.cast(_height, tf.float32)
    _width_f  = tf.cast(_width,  tf.float32)

    _wrap_mode = wrap_mode

    output = _transform(input_images, x_offset)
    return output

##SSIM 
def SSIM(imagex,imagey,max_value=1):
    c1 = (0.01*max_value)**2
    c2 = (0.01*max_value)**2
    mux = tf.nn.avg_pool2d(imagex,(3,3),(1,1),padding='SAME')
    muy = tf.nn.avg_pool2d(imagey,(3,3),(1,1),padding='SAME')
    
    varx = tf.nn.avg_pool2d(imagex**2,(3,3),(1,1),padding='SAME')-mux**2
    vary = tf.nn.avg_pool2d(imagey**2,(3,3),(1,1),padding='SAME')-muy**2
    
    sigmaxy = tf.nn.avg_pool2d(imagex*imagey,(3,3),(1,1),padding='SAME')-mux*muy
    
    
    SSIM_n = (2 * mux * muy + c1) * (2 * sigmaxy + c2)
    SSIM_d = (mux ** 2 + muy ** 2 + c1) * (varx + vary + c2)

    SSIM = SSIM_n / SSIM_d

    return tf.clip_by_value((1 - SSIM) / 2, 0, 1)

##image generation
def generate_image_left(img, disp):
        return bilinear_sampler_1d_h(img, -disp)
    
def generate_image_right(img, disp):
        return bilinear_sampler_1d_h(img, disp)

##depth estimation loss spatial 
def dept_est_loss(depth_prediction_left,depth_prediction_right,left_image,right_image):
    B = 10 #base length (cm)
    f = .04 #focal length (cm)
    
    #there is the 4 disparity maps 2 for each stero image (i.e right stereo image and left stereo image)
    ##k and k+1 disparities for left image
    ##left image generates left disparity 
    left_disparity_k = depth_prediction_left[:,:,:,0] #(B*f)/depth_prediction_left[:,:,:,0] currently assuming the depth_prediction as disparity
    left_disp_batch_num  = depth_prediction_left.shape[0]
    left_disp_height = depth_prediction_left.shape[1]
    left_disp_width = depth_prediction_left.shape[2]
    left_disparity_k = tf.reshape(left_disparity_k,[left_disp_batch_num,left_disp_height,left_disp_width,1])
    #print(left_disparity_k.shape)
    #tf.print(left_disparity_k)
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
 
