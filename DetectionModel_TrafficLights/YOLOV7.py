#!/usr/bin/env python
# coding: utf-8

# In[1]:

import random
import torch
import cv2
import os
import numpy as np


# In[3]:


def load_images_from_folder(path,hsv=False,hls=False):
    images=[]; Paths=[]
    for file in os.listdir(path):
        Paths.append(file)
        img = cv2.imread(os.path.join(path,file))     
        images.append(img)
    return np.array(images),Paths

def filter_color(image,min,max):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, min, max)
    masked = cv2.bitwise_and(hsv, hsv, mask=cv2.bitwise_not(mask))
    reconstructed = cv2.cvtColor(masked, cv2.COLOR_HSV2BGR)
    return reconstructed

def plot_one_box(x, img, color=None, label=None, line_thickness=3):
    # Plots one bounding box on image img
    tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [0, 0, 0], thickness=tf, lineType=cv2.LINE_AA)
        
def get_roi(img):
    img[:,:img.shape[1]//4]=0;img[:,3*(img.shape[1]//4):]=0
    return img

def get_traffic_lights_rois(hsv):
    top=hsv[:hsv.shape[0]//3]
    middle=hsv[hsv.shape[0]//3:2*(hsv.shape[0]//3)]
    bottom=hsv[2*(hsv.shape[0]//3):]
    return top,middle,bottom

def get_bounding_box_area(det):
    return (int(det[2])-int(det[0]))*int((det[3])-int(det[1]))

def bgr_to_hsv(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

def crop_to_detection(hsv,det):
    start_point = int(det[0]),int(det[1])
    end_point = int(det[2]),int(det[3])
    hsv=hsv[start_point[1]:end_point[1],start_point[0]:end_point[0]]
    return hsv

def get_mask(img,mini,maxi):
    return cv2.inRange(img,mini,maxi)

def count_extracted_colored_pixels(mask):
    return cv2.countNonZero(mask)
    
def get_label(red,green,threshold):
    maxi = max(red,green)
    if maxi<threshold:
        name="skip"
    else:
        name = "red" if red>green else "green"
    return name

def save_img(path,img):
    cv2.imwrite(path,img)
    
def load_model():
    return torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

def detect(model,img,name,save):
    
    cpy=img.copy()
    
    ## is red ? 
    is_red = False

    #### REMOVE THE LEFT AND RIGHT QUARTERS ####
    img = get_roi(img)

    detections = model(img).pred[0]
    for det in detections:
        area = get_bounding_box_area(det)
        *xyxy, conf, cls = det

        #### ONLY HANDLE TRAFFIC LIGHTS AND NEGELCT SMALL BOUNDING BOXES ####
        if cls != 9 or area<500:
            continue

        hsv = bgr_to_hsv(img)

        ##### CROP WHOLE IMAGE TO DETECTION ####
        hsv = crop_to_detection(hsv,det)

        ################# TRAFFIC LIGHTS HADNLING ############################

        #RED
        #YELLOW
        #GREEN

        ## CROPPING THE DETECTION TO 3 PARTS ##
        top,middle,bottom = get_traffic_lights_rois(hsv)

        #### GET GREEN FROM THE BOTTOM PART ####
        mask_green = get_mask(bottom,(40,40,40),(70,255,255))

        #### GET RED FROM THE TOP PART ####
        mask_red = get_mask(top, (0,80,20), (12,255,255))

        #### COUNT NUMBER OF PIXELS ####
        red=count_extracted_colored_pixels(mask_red)
        green=count_extracted_colored_pixels(mask_green)

        #### DECIDE ON THE LABEL ####
        label = get_label(red,green,20)
        
        if label=='red':
            is_red=True

        #### PLOT THE BOX ####
        plot_one_box(xyxy,cpy,label=label,color=(255,255,255),line_thickness=1)

    #### SAVE THE DETECTIONS ####
    if save:
        save_img(f'output/{name}.png',cpy)
    return cpy,is_red







