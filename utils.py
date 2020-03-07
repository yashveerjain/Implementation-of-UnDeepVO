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


def euler2mat(z, y, x):
  """Converts euler angles to rotation matrix
   TODO: remove the dimension for 'N' (deprecated for converting all source
         poses altogether)
   Reference: https://github.com/pulkitag/pycaffe-utils/blob/master/rot_utils.py#L174
  Args:
      z: rotation angle along z axis (in radians) -- size = [B, N]
      y: rotation angle along y axis (in radians) -- size = [B, N]
      x: rotation angle along x axis (in radians) -- size = [B, N]
  Returns:
      Rotation matrix corresponding to the euler angles -- size = [B, N, 3, 3]
  """
  B = tf.shape(z)[0]
  N = 1
  z = tf.clip_by_value(z, -np.pi, np.pi)
  y = tf.clip_by_value(y, -np.pi, np.pi)
  x = tf.clip_by_value(x, -np.pi, np.pi)

  # Expand to B x N x 1 x 1
  z = tf.expand_dims(tf.expand_dims(z, -1), -1)
  y = tf.expand_dims(tf.expand_dims(y, -1), -1)
  x = tf.expand_dims(tf.expand_dims(x, -1), -1)

  zeros = tf.zeros([B, N, 1, 1])
  ones  = tf.ones([B, N, 1, 1])

  cosz = tf.cos(z)
  sinz = tf.sin(z)
  rotz_1 = tf.concat([cosz, -sinz, zeros], axis=3)
  rotz_2 = tf.concat([sinz,  cosz, zeros], axis=3)
  rotz_3 = tf.concat([zeros, zeros, ones], axis=3)
  zmat = tf.concat([rotz_1, rotz_2, rotz_3], axis=2)

  cosy = tf.cos(y)
  siny = tf.sin(y)
  roty_1 = tf.concat([cosy, zeros, siny], axis=3)
  roty_2 = tf.concat([zeros, ones, zeros], axis=3)
  roty_3 = tf.concat([-siny,zeros, cosy], axis=3)
  ymat = tf.concat([roty_1, roty_2, roty_3], axis=2)

  cosx = tf.cos(x)
  sinx = tf.sin(x)
  rotx_1 = tf.concat([ones, zeros, zeros], axis=3)
  rotx_2 = tf.concat([zeros, cosx, -sinx], axis=3)
  rotx_3 = tf.concat([zeros, sinx, cosx], axis=3)
  xmat = tf.concat([rotx_1, rotx_2, rotx_3], axis=2)

  rotMat = tf.matmul(tf.matmul(xmat, ymat), zmat)
  return rotMat

def pose_vec2mat(translation,rotation):
  """Converts 6DoF parameters to transformation matrix
  Args:
      vec: 6DoF parameters in the order of tx, ty, tz, rx, ry, rz -- [B, 6]
  Returns:
      A transformation matrix -- [B, 4, 4]
  """
  batch_size, _ = translation.get_shape().as_list()
  translation = tf.expand_dims(translation, -1)
  rx = tf.slice(rotation, [0, 0], [-1, 1])
  ry = tf.slice(rotation, [0, 1], [-1, 1])
  rz = tf.slice(rotation, [0, 2], [-1, 1])
  rot_mat = euler2mat(rz, ry, rx)
  rot_mat = tf.squeeze(rot_mat, axis=[1])
  filler = tf.constant([0.0, 0.0, 0.0, 1.0], shape=[1, 1, 4])
  filler = tf.tile(filler, [batch_size, 1, 1])
  transform_mat = tf.concat([rot_mat, translation], axis=2)
  transform_mat = tf.concat([transform_mat, filler], axis=1)
  return transform_mat

def pixel2cam(depth, pixel_coords, intrinsics, is_homogeneous=True):
  """Transforms coordinates in the pixel frame to the camera frame.
  Args:
    depth: [batch, height, width]
    pixel_coords: homogeneous pixel coordinates [batch, 3, height, width]
    intrinsics: camera intrinsics [batch, 3, 3]
    is_homogeneous: return in homogeneous coordinates
  Returns:
    Coords in the camera frame [batch, 3 (4 if homogeneous), height, width]
  """
  batch, height, width = depth.get_shape().as_list()
  depth = tf.reshape(depth, [batch, 1, -1])
  pixel_coords = tf.reshape(pixel_coords, [batch, 3, -1])
  cam_coords = tf.matmul(tf.matrix_inverse(intrinsics), pixel_coords) * depth
  if is_homogeneous:
    ones = tf.ones([batch, 1, height*width])
    cam_coords = tf.concat([cam_coords, ones], axis=1)
  cam_coords = tf.reshape(cam_coords, [batch, -1, height, width])
  return cam_coords

def cam2pixel(cam_coords, proj):
  """Transforms coordinates in a camera frame to the pixel frame.
  Args:
    cam_coords: [batch, 4, height, width]
    proj: [batch, 4, 4]
  Returns:
    Pixel coordinates projected from the camera frame [batch, height, width, 2]
  """
  batch, _, height, width = cam_coords.get_shape().as_list()
  cam_coords = tf.reshape(cam_coords, [batch, 4, -1])
  unnormalized_pixel_coords = tf.matmul(proj, cam_coords)
  x_u = tf.slice(unnormalized_pixel_coords, [0, 0, 0], [-1, 1, -1])
  y_u = tf.slice(unnormalized_pixel_coords, [0, 1, 0], [-1, 1, -1])
  z_u = tf.slice(unnormalized_pixel_coords, [0, 2, 0], [-1, 1, -1])
  x_n = x_u / (z_u + 1e-10)
  y_n = y_u / (z_u + 1e-10)
  pixel_coords = tf.concat([x_n, y_n], axis=1)
  pixel_coords = tf.reshape(pixel_coords, [batch, 2, height, width])
  return tf.transpose(pixel_coords, perm=[0, 2, 3, 1])

def meshgrid(batch, height, width, is_homogeneous=True):        
      """Construct a 2D meshgrid.
      Args:
        batch: batch size
        height: height of the grid
        width: width of the grid
        is_homogeneous: whether to return in homogeneous coordinates
      Returns:
        x,y grid coordinates [batch, 2 (3 if homogeneous), height, width]
      """
      """
      x_t = tf.matmul(tf.ones(shape=tf.stack([height, 1])),
                      tf.transpose(tf.expand_dims(
                          tf.linspace(-1.0, 1.0, width), 1), [1, 0]))
      y_t = tf.matmul(tf.expand_dims(tf.linspace(-1.0, 1.0, height), 1),
                      tf.ones(shape=tf.stack([1, width])))
      x_t = (x_t + 1.0) * 0.5 * tf.cast(width - 1, tf.float32)
      y_t = (y_t + 1.0) * 0.5 * tf.cast(height - 1, tf.float32)
      """
      width_f = tf.cast(width,tf.float32)
      height_f = tf.cast(height,tf.float32)
      x_t,y_t = tf.meshgrid(tf.linspace(0.0,width_f,width),tf.linspace(0.0,height_f,height))
      if is_homogeneous:
        ones = tf.ones_like(x_t)
        coords = tf.stack([x_t, y_t, ones], axis=0)
      else:
        coords = tf.stack([x_t, y_t], axis=0)
      coords = tf.tile(tf.expand_dims(coords, 0), [batch, 1, 1, 1])
      return coords

def projective_inverse_warp(img, depth, pose, intrinsics):
  """Inverse warp a source image to the target image plane based on projection.
  Args:
    img: the source image [batch, height_s, width_s, 3]
    depth: depth map of the target image [batch, height_t, width_t]
    pose: target to source camera transformation matrix [batch, 6], in the
          order of tx, ty, tz, rx, ry, rz
    intrinsics: camera intrinsics [batch, 3, 3]
  Returns:
    Source image inverse warped to the target image plane [batch, height_t,
    width_t, 3]
  """
  batch, height, width, _ = img.get_shape().as_list()
  # Convert pose vector to matrix
  pose = pose_vec2mat(pose)
  # Construct pixel grid coordinates
  pixel_coords = meshgrid(batch, height, width)
  # Convert pixel coordinates to the camera frame
  cam_coords = pixel2cam(depth, pixel_coords, intrinsics)
  # Construct a 4x4 intrinsic matrix (TODO: can it be 3x4?)
  filler = tf.constant([0.0, 0.0, 0.0, 1.0], shape=[1, 1, 4])
  filler = tf.tile(filler, [batch, 1, 1])
  intrinsics = tf.concat([intrinsics, tf.zeros([batch, 3, 1])], axis=2)
  intrinsics = tf.concat([intrinsics, filler], axis=1)
  # Get a 4x4 transformation matrix from 'target' camera frame to 'source'
  # pixel frame.
  proj_tgt_cam_to_src_pixel = tf.matmul(intrinsics, pose)
  src_pixel_coords = cam2pixel(cam_coords, proj_tgt_cam_to_src_pixel)
  output_img = bilinear_sampler_1d_h(img, src_pixel_coords)
  return output_img 
