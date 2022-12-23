#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 18-5-23 上午11:33
# @Author  : MaybeShewill-CV
# @Site    : https://github.com/MaybeShewill-CV/lanenet-lane-detection
# @File    : test_lanenet.py
# @IDE: PyCharm Community Edition
"""
test LaneNet model on single image
"""
import argparse
import os.path as ops
import time
import math
import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow.compat.v1 as tf
import sys
import os
myDir = os.getcwd()
sys.path.append(myDir)
from pathlib import Path
path = Path(myDir)
a=str(path.parent.absolute())
sys.path.append(a)
from lanenet_model import lanenet
from lanenet_model import lanenet_postprocess
from local_utils.config_utils import parse_config_utils
from local_utils.log_util import init_logger

CFG = parse_config_utils.lanenet_cfg
LOG = init_logger.get_logger(log_file_name_prefix='lanenet_test')


def init_args():
    """

    :return:
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_path', type=str, help='The image path or the src image save dir')
    parser.add_argument('--weights_path', type=str, help='The model weights path')
    parser.add_argument('--with_lane_fit', type=args_str2bool, help='If need to do lane fit', default=True)

    return parser.parse_args()


def args_str2bool(arg_value):
    """

    :param arg_value:
    :return:
    """
    if arg_value.lower() in ('yes', 'true', 't', 'y', '1'):
        return True

    elif arg_value.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Unsupported value encountered.')


def minmax_scale(input_arr):
    """

    :param input_arr:
    :return:
    """
    min_val = np.min(input_arr)
    max_val = np.max(input_arr)

    output_arr = (input_arr - min_val) * 255.0 / (max_val - min_val)

    return output_arr


def test_lanenet(image_path, weights_path, with_lane_fit=True):
    """

    :param image_path:
    :param weights_path:
    :param with_lane_fit:
    :return:
    """
    assert ops.exists(image_path), '{:s} not exist'.format(image_path)

    LOG.info('Start reading image and preprocessing')
    t_start = time.time()
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    img_center_lane = image
    image_vis = image
    image = cv2.resize(image, (512, 256), interpolation=cv2.INTER_LINEAR)
    image = image / 127.5 - 1.0
    LOG.info('Image load complete, cost time: {:.5f}s'.format(time.time() - t_start))
    tf.disable_eager_execution()
    input_tensor = tf.placeholder(dtype=tf.float32, shape=[1, 256, 512, 3], name='input_tensor')

    net = lanenet.LaneNet(phase='test', cfg=CFG)
    binary_seg_ret, instance_seg_ret = net.inference(input_tensor=input_tensor, name='LaneNet')

    postprocessor = lanenet_postprocess.LaneNetPostProcessor(cfg=CFG)

    # Set sess configuration
    sess_config = tf.ConfigProto()
    sess_config.gpu_options.per_process_gpu_memory_fraction = CFG.GPU.GPU_MEMORY_FRACTION
    sess_config.gpu_options.allow_growth = CFG.GPU.TF_ALLOW_GROWTH
    sess_config.gpu_options.allocator_type = 'BFC'

    sess = tf.Session(config=sess_config)

    # define moving average version of the learned variables for eval
    with tf.variable_scope(name_or_scope='moving_avg'):
        variable_averages = tf.train.ExponentialMovingAverage(
            CFG.SOLVER.MOVING_AVE_DECAY)
        variables_to_restore = variable_averages.variables_to_restore()

    # define saver
    saver = tf.train.Saver(variables_to_restore)

    with sess.as_default():
        saver.restore(sess=sess, save_path=weights_path)

        t_start = time.time()
        loop_times = 500
        for i in range(loop_times):
            binary_seg_image, instance_seg_image = sess.run(
                [binary_seg_ret, instance_seg_ret],
                feed_dict={input_tensor: [image]}
            )
        t_cost = time.time() - t_start
        t_cost /= loop_times
        LOG.info('Single imgae inference cost time: {:.5f}s'.format(t_cost))

        postprocess_result = postprocessor.postprocess(
            binary_seg_result=binary_seg_image[0],
            instance_seg_result=instance_seg_image[0],
            source_image=image_vis,
            with_lane_fit=with_lane_fit,
            data_source='tusimple'
        )
        mask_image = postprocess_result['mask_image']
        if with_lane_fit:
            lane_params = postprocess_result['fit_params']
            LOG.info('Model have fitted {:d} lanes'.format(len(lane_params)))
            for i in range(len(lane_params)):
                LOG.info('Fitted 2-order lane {:d} curve param: {}'.format(i + 1, lane_params[i]))

        for i in range(CFG.MODEL.EMBEDDING_FEATS_DIMS):
            instance_seg_image[0][:, :, i] = minmax_scale(instance_seg_image[0][:, :, i])
        embedding_image = np.array(instance_seg_image[0], np.uint8)

        # plt.figure('mask_image')
        # plt.imshow(mask_image[:, :, (2, 1, 0)])
        # plt.figure('src_image')
        # plt.imshow(image_vis[:, :, (2, 1, 0)])
        # plt.figure('instance_image')
        # plt.imshow(embedding_image[:, :, (2, 1, 0)])
        plt.figure('binary_image')
        plt.imshow(binary_seg_image[0] * 255, cmap='gray')
        plt.show()

        image = (binary_seg_image[0]*255).astype(np.uint8)
        # cv2.imshow('mat', image)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        sliding_window(image, image_vis, (15,50))
        

    sess.close()

    return
    
def PerspectiveTransform(src,dst):
    mat = cv2.getPerspectiveTransform(src,dst)
    mat_inv = cv2.getPerspectiveTransform(dst,src)
    return mat,mat_inv

def warpPerspective(img, mat, size):
    return cv2.warpPerspective(img, mat,size)    


def sliding_window(img,dst_colored, window_size):
    #window_size = (height , width)
    #shape= (720 * 1280)
    #convert to colored img to draw colored line and windows on top of it
    # for prespective trasnform points eg input_top_left = [width, height]
    size = [img.shape[1], img.shape[0]]

    

    input_top_left = [188,110]
    input_top_right = [312,110]
    input_bottom_right = [506,219]
    input_bottom_left = [55,219]

    src_pt = np.float32([input_bottom_left,input_top_left,input_top_right,input_bottom_right])
    dst_pt = np.float32([[0,size[1]],[0,0],[size[0],0],[size[0],size[1]]])

    M,Minv = PerspectiveTransform(src_pt , dst_pt)
    
                
#         dst_colored = perspective_warp(frame ,src=input_points , dst=p2)
    dst_colored = warpPerspective(img ,M , size)
    img = warpPerspective(img ,M , size)

    out_img = cv2.cvtColor(img , cv2.COLOR_GRAY2RGB)
    
    nwindows = int(img.shape[0] / window_size[0])
   
    # find peaks of left and right lanes
    histogram = np.sum(img, axis=0)
    midpoint = int(histogram.shape[0]//2)
    start_left_x= np.argmax(histogram[:midpoint]) -30
    start_right_x = np.argmax(histogram[midpoint + 100:]) + midpoint +100
    
    #get positions of white pixels in original img
    white_pixels = img.nonzero()
    # print(white_pixels)
    white_x = np.array(white_pixels[1])
    
    white_y = np.array(white_pixels[0])

    
    # the left and right lane indices that we are going to find
    left_lane_indices = []
    right_lane_indices = []
    
    for window in range(nwindows):
        
        # find the boundary of each window
        win_bot = img.shape[0] - (window+1)*window_size[0]
        win_top = img.shape[0] - window*window_size[0]
        left_lane_lbound = start_left_x - window_size[1]//2
        left_lane_rbound = start_left_x + window_size[1]//2
        right_lane_lbound = start_right_x - window_size[1]//2
        right_lane_rbound = start_right_x + window_size[1]//2
        
        #draw the windows in red
        cv2.rectangle(dst_colored,(left_lane_lbound,win_bot),(left_lane_rbound,win_top),(255,0,0), 3) 
        cv2.rectangle(dst_colored,(right_lane_lbound,win_bot),(right_lane_rbound,win_top),(255,0,0), 3) 
        
        #locate the white pixels that lie within current window 
        good_left_inds = ((white_y >= win_bot) & (white_y < win_top) & 
        (white_x >= left_lane_lbound) &  (white_x < left_lane_rbound)).nonzero()[0]
        good_right_inds = ((white_y >= win_bot) & (white_y < win_top) & 
        (white_x >= right_lane_lbound) &  (white_x < right_lane_rbound)).nonzero()[0]
        
        left_lane_indices.append(good_left_inds)
        right_lane_indices.append(good_right_inds)
        
        #if the window contain black pixels don't shift it
        if len(good_left_inds) > 0:
            start_left_x = int(np.mean(white_x[good_left_inds]))
            
        if len(good_right_inds) > 0:        
           
            
            start_right_x = int(np.mean(white_x[good_right_inds]))

            
    left_lane_indices = np.concatenate(left_lane_indices)
    right_lane_indices = np.concatenate(right_lane_indices)

    
    leftx = white_x[left_lane_indices]
    lefty = white_y[left_lane_indices] 
    rightx = white_x[right_lane_indices]
    righty = white_y[right_lane_indices] 
    
    #fit a 2nd degree curve to the white pixels positions we found
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)
    
    
    # #predict the current position
    # ploty = np.linspace(0, 719, 720 )
    # left_fitx = np.polyval(left_fit , ploty)
    # right_fitx = np.polyval(right_fit , ploty)
    
    # left_poly = np.asarray(tuple(zip(left_fitx,ploty)) ,np.int32)
    # right_poly = np.asarray(tuple(zip(right_fitx,ploty)),np.int32)
    
    steering_angle_calc(dst_colored, [right_fit, left_fit], np.zeros_like(dst_colored), M)
    
    return 


def steering_angle_calc(image_vis, lane_params, img_center_lane, M):
        # Trying to calculate the steering angle required to help car stay in centre of lane
        # Assuming car centre is in centre of the image
        # image shape is (height, width, channels)
        # In implementation of lane fitting x (input) is the height and y (output) is the width
        # cv2.cvtColor(img_center_lane, cv2.COLOR_GRAY2RGB)  

        dist_reversed = 160
        dist = image_vis.shape[0] - dist_reversed                                                     # this is the target distance after which we want to make steering angle calculations
        car_center = (image_vis.shape[1]//2, image_vis.shape[0])                         # This is the car's centre assuming the car is always in the centre of the frame
        center_after_dist = [car_center[0], dist]                        # the car's centre after the target distance if it continued moving in a straight line

        #transforming centre coordinates into bird-eye view
        # px = (M[0][0]*center_after_dist[0] + M[0][1]*center_after_dist[1] + M[0][2]) / ((M[2][0]*center_after_dist[0] + M[2][1]*center_after_dist[1] + M[2][2]))
        # py = (M[1][0]*center_after_dist[0] + M[1][1]*center_after_dist[1] + M[1][2]) / ((M[2][0]*center_after_dist[0] + M[2][1]*center_after_dist[1] + M[2][2]))
        

        # center_after_dist = (int(px), int(py))
        
        center_after_dist = cv2.perspectiveTransform(np.array([[center_after_dist]], dtype=np.float32), M)[0][0]
        print(center_after_dist)

        ploty = np.linspace(0, image_vis.shape[0]-1, image_vis.shape[0] )                                # values of height pixels from which to get width values
        lane1_pts_x = lane_params[0][0] * ploty **2 + lane_params[0][1] * ploty + lane_params[0][2]
        
        lane2_pts_x = np.polyval( lane_params[1], ploty )

        lane1_pts = np.asarray( tuple(zip(lane1_pts_x, ploty )), dtype = np.int32)
        lane2_pts = np.asarray( tuple(zip(lane2_pts_x, ploty )), dtype = np.int32)
        # print(lane1_pts)
        cv2.polylines(img_center_lane , [lane1_pts] , isClosed = False , color=(255,0,0) , thickness=10)
        cv2.polylines(img_center_lane , [lane2_pts] , isClosed = False , color=(255,0,0) , thickness=10)

        centerlane_pts = np.asarray((lane1_pts + lane2_pts) / 2, dtype= np.int32)
        
        
        centerlane_after_dist = centerlane_pts[dist]
        

        opposite_dist = np.abs(centerlane_after_dist[0] - center_after_dist[0])

        steering_angle = np.arctan(opposite_dist/dist_reversed)
        steering_angle = math.degrees(steering_angle)

        # print(f"The center lane points are: {centerlane_pts}")
        cv2.polylines(img_center_lane , [centerlane_pts] , isClosed = False , color=(255,0,0) , thickness=10)   

        cv2.imshow('Sliding window image', image_vis)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        cv2.imshow('center lane image', img_center_lane)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        if centerlane_after_dist[0] < center_after_dist[0]:
            steering_angle *=-1
        # steering_angle/=90
        print(f"The steering angle is: {steering_angle}")


        return steering_angle

if __name__ == '__main__':
    """
    test code
    """
    # init args
    args = init_args()

    test_lanenet(args.image_path, args.weights_path, with_lane_fit=args.with_lane_fit)
