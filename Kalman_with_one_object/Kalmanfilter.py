import random
import torch
import cv2
import numpy as np
import filterpy
from filterpy.kalman import KalmanFilter
from filterpy.common import Q_discrete_white_noise


def yolobbox2bbox(x,y,w,h):
    x1, y1 = x-w/2, y-h/2
    x2, y2 = x+w/2, y+h/2
    return x1, y1, x2, y2

def extract_frames(videoPath):
    frames = []
    video = cv2.VideoCapture(videoPath)
    while True:
        read, frame= video.read()
        if not read:
            break
        frames.append(frame)
    return np.array(frames)

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

def save_img(path,img):
    cv2.imwrite(path,img)

def get_bounding_box_area(det):
    return (int(det[2])-int(det[0]))*int((det[3])-int(det[1]))

def load_model():
    return torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)


def predict_BB(img,model,filter):
    detections =model(img).xywh[0]
    for det in detections:

        *xyxy, conf, cls = det
        
        filter.predict()
        filter.update(xyxy)
        xyxy=yolobbox2bbox(*xyxy)
        plot_one_box(xyxy,img,color=(255,0,0),line_thickness=1)
        filter_output=filter.x[:4]
        plot_one_box(yolobbox2bbox(*filter_output),img,color=(0,255,0),line_thickness=1)
    
    

def kalmanInit():
    danny_filta= KalmanFilter (dim_x=6, dim_z=4)
# dim_x = 6 ? x,y,w,h,vx,vy ?
#initial value for the state vector
#                           x,y,w,h,vx,vy
    danny_filta.x = np.array([
    [2.], 
    [0.],
    [3],
    [4],
    [3]
    ,[4]])   
    #state transtion matrix :  what intialization?

    danny_filta.F = np.array([[1.,0.,0.,0.,0.1,0.],
                            [0,1.,0.,0.,0,0.1],
                            [0,0,1.,0.,0.,0. ],
                            [0,0,0,1.,0.,0. ],
                            [0,0,0,0,1,0   ],
                            [0,0,0,0,0,1.  ]
                            ])
    # measurement matrix 
    # transorm from state space to measurment space 
    # (4,1) ? because we need x,y,w,h

    danny_filta.H = np.array([[1.,0.,0,0,0,0],
                            [0,1,0,0,0,0],          
                            [0,0.,1,0,0,0],         
                            [0,0.,0,1,0,0]])

    # covariance matrix 
    # need to check how we can intialize it 
    # dimx * dimx
    danny_filta.P = np.array([[1000., 0.,0,0,0,0],
                            [0., 1000.,0,0,0,0],
                            [0., 0.,1000,0,0,0],
                            [0., 0.,0,10000,0,0],
                            [0., 0.,0,0,1000,0],
                            [0., 0.,0,0,0,1000], ])

    # measurement noise
    # shape : (dim_z,dim_z)
    danny_filta.R = np.array([[5,0,0,0],
    [0,5,0,0],
    [0,0,5,0],
    [0,0,0,5]])

    # process noise 
    # dimx * dimx
    danny_filta.Q=np.zeros((6,6))
    return danny_filta




f=kalmanInit()
model =load_model()
model.classes=0
frame_array=extract_frames('danny.mp4')
predict_BB(frame_array,model ,f)



