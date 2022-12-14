#!/usr/bin/env python
# coding: utf-8

# In[1]:




import carla
import math
import random
import cv2
import numpy as np
import glob # for getting images from camera
camera_path = "D:/Handsa/carla/camera_images"
import matplotlib.pyplot as plt
# import YOLOV7
import keyboard
import os
import Phase1
from Phase1 import calc_off_dist, sliding_window, detect_edges, PerspectiveTransform, warpPerspective, debug_mode


# # function setpath fe el traffic manager btkhaly el vehicle tfollow certain locations t3addy 3leha

# # lw 7sal moshkla fe el script sheel goz2 el sync fe el traffic manager

# # connectig client to server and getting access to world

# In[2]:

#***********************************************************************Client and World Initialization****************************************************************
def init_client_world(port):
    client = carla.Client('localhost', port) # if client is another machine rathr than host, insert ipaddress instead of localhost
    client.set_timeout(50000)
    world = client.get_world() # getting access to world traaffic, trees, etc and manipulate them
    
    return client, world

# Sync world with client
def sync_world(world):
    settings = world.get_settings()
    settings.synchronous_mode = True 
    settings.fixed_delta_seconds = 0.05
    world.apply_settings(settings)

# Desync world before stopping client
def desync_world(world):
    settings = world.get_settings()
    settings.synchronous_mode = False 
    world.apply_settings(settings)
#***********************************************************************************************************************************************************************






# # get access to buleprints and spawn points in carla

# In[3]:


def get_bps_spawn_pts(world):
    blue_prints = world.get_blueprint_library()
    spawn_pts = world.get_map().get_spawn_points() # get access to spawn points to spawn objects into street
    return blue_prints, spawn_pts







# In[4]:

#********************************************************************Vehicle Functions for control and spawning*********************************************************
def spawn_vehicle(blue_prints, spawn_pts, world):
    vehicle_bp = blue_prints.find("vehicle.audi.tt")
    vehicle = world.spawn_actor(vehicle_bp, random.choice(spawn_pts))
    return vehicle






def spawn_vehicle_follow(blue_prints, waypoint,vehicle, world):
     
     return world.spawn_actor(blue_prints.find("vehicle.audi.tt"), transform =carla.Transform(waypoint.transform.location + carla.Location(x = 30, z=2.5), vehicle.get_transform().rotation ))





def stop_vehicle(vehicle):

    ctrl = vehicle.get_control()
    ctrl.brake = 1.0
    ctrl.throttle = 0.0
    ctrl.hand_brake = True
    vehicle.apply_control(ctrl)
     



def automatic_control(vehicle):
    
    # code for setting wheel physics:
    front_left_wheel  = carla.WheelPhysicsControl(tire_friction=2.0, damping_rate=1.5, max_steer_angle=70.0, long_stiff_value=1000)
    front_right_wheel = carla.WheelPhysicsControl(tire_friction=2.0, damping_rate=1.5, max_steer_angle=70.0, long_stiff_value=1000)
    rear_left_wheel   = carla.WheelPhysicsControl(tire_friction=3.0, damping_rate=1.5, max_steer_angle=0.0,  long_stiff_value=1000)
    rear_right_wheel  = carla.WheelPhysicsControl(tire_friction=3.0, damping_rate=1.5, max_steer_angle=0.0,  long_stiff_value=1000)
    
    # combining wheels in one list
    wheels = [front_left_wheel, front_right_wheel, rear_left_wheel, rear_right_wheel]
    
    # code for changing physics control of vehicle
    
    physics_control = vehicle.get_physics_control()

    physics_control.torque_curve = [carla.Vector2D(x=0, y=400), carla.Vector2D(x=1300, y=600)]
    physics_control.max_rpm = 10000
    physics_control.moi = 1.0
    physics_control.damping_rate_full_throttle = 0.0
    physics_control.use_gear_autobox = True
    physics_control.gear_switch_time = 0.5
    physics_control.clutch_strength = 10
    physics_control.mass = 10000
    physics_control.drag_coefficient = 0.25
    physics_control.steering_curve = [carla.Vector2D(x=0, y=1), carla.Vector2D(x=100, y=1), carla.Vector2D(x=300, y=1)]
    physics_control.use_sweep_wheel_collision = True
    physics_control.wheels = wheels
    
    # apply physics control on vehicle
    vehicle.apply_physics_control(physics_control)




def enable_auto_pilot(client, port, vehicles_list, world):
    
    tm = client.get_trafficmanager(port)
    tm_port = tm.get_port()
    # setting autopilot for vehciles in vehicles_list
    for v in vehicles_list:
        v.set_autopilot(True, tm_port)
        # tm.ignore_vehicles_percentage(v, 100)
        tm.keep_right_rule_percentage(v, 100)
        tm.ignore_lights_percentage(v,100)
        
        
        
    # traffic manager should be run in sync mode with server
    # here is how it is done
    
    # Set the simulation to sync mode
    # init_settings = world.get_settings()
    # settings = world.get_settings()
    # settings.synchronous_mode = True
    
    # After that, set the TM to sync mode
    tm.set_synchronous_mode(True)


    # Tick the world in the same client
    # world.apply_settings(init_settings)
    # world.tick()



#************************************************************************************************************************************************************************









#******************************************************************Camera Functions**************************************************************************************
def spawn_camera(blue_print, vehicle, world ,path):
    camera_bp = blue_print.find('sensor.camera.rgb') # getting bluprint of sensor
    camera_init_trans = carla.Transform(carla.Location(z=2))
    camera = world.try_spawn_actor(camera_bp , camera_init_trans, attach_to = vehicle, attachment_type = carla.AttachmentType.Rigid)
#     camera.listen(lambda image: image.save_to_disk(path+"/%.6d.png" % image.frame)) # saving camera sensor output
    
    return camera

def call_back_camera_yolo(model, img, label):
    init_shape = (img.height, img.width, 4)
    image = np.array(img.raw_data).reshape(init_shape)[:,:,:3]

    detect_img, label[0] = YOLOV7.detect(model, image,img.frame, True)


    cv2.imshow('img',detect_img)
    _ = cv2.waitKey(1)

def call_back_camera(img,out_img, save):
    init_shape = (img.height, img.width, 4)
    out_img[0] = np.array(img.raw_data).reshape(init_shape)[:,:,:3]
    
    
    
    
    if save:
        # path = os.path.join('output', f'{img.frame}.png' )
        path = '/home/amr/GradProj/output/'+f'{img.frame}.png'
        # img.save_to_disk(path)
        print(path)
        cv2.imwrite(path, out_img[0])
    
def call_back_camera_lane(img,out_img,src_pts, dst_pts, s_thresh, l_thresh, shad_thresh,old_list,out_src, out_detect, save):

    init_shape = (img.height, img.width, 4)
    image = np.array(img.raw_data).reshape(init_shape)[:,:,:3]
    out_src.write(image)
    out_img[0],old_list[0], old_list[1] = lane_detection_output(image, src_pts, dst_pts, s_thresh, l_thresh, shad_thresh,old_list[0],old_list[1],size = (800, 600), debug = 0)
    # print(f"right poly old after call back is: {old_list[0]}\n")
    out_detect.write(out_img[0])
    
    if save:
        path = '/home/amr/GradProj/output/'+ f'{img.frame}.png'
        # img.save_to_disk(path)
        
        cv2.imwrite(path, image)

        

# getting images from camera sensor saved on disk
def get_imgs(path):
   images = []
   for img in glob.glob(path+"/*.png"):
       image = cv2.imread(img)
       images.append(image)
   return images
    
#***********************************************************************************************************************************************************************









#*****************************************************************GPS functions for spawning and retrieving sensor data*************************************************

def store_coord(coord1,gps_coord):
    # longitude is the first element in list,
    # latitude is the second element in list
    coord1[1] = gps_coord.latitude
    coord1[0] = gps_coord.longitude


def spawn_gps(blue_print, vehicle, world, offset=1):
    gps_bp = blue_print.find('sensor.other.gnss')
    gps_init_trans = carla.Transform(carla.Location(x= offset, z=2.5))
    gps = world.try_spawn_actor(gps_bp, gps_init_trans, attach_to = vehicle)
    
    # gps.listen(lambda gps_coord: store_coord(latitude=latitude, longitude= longitude, gps_coord=gps_coord))

    return gps

# def spawn_gps_2(blue_print, vehicle, world):

#     gps_bp = blue_print.find('sensor.other.gnss')
#     gps_transform = carla.Transform(carla.Location(x = offset, z = 2.5))
#     gps = world.spawn_actor(gps_bp, gps_transform, attach_to = vehicle)

#     return gps


def spawn_gps_err(blue_print, vehicle, world,offset = 1, acc= 0):

    gps_bps = blue_print.find('sensor.other.gnss')
    # setting accuracy of latitude
    gps_bps.set_attribute('noise_lat_bias', '0')
    gps_bps.set_attribute('noise_lat_stddev', str(math.sqrt(acc)))
    # setting accuracy of longitude
    gps_bps.set_attribute('noise_lon_bias', '0')
    gps_bps.set_attribute('noise_lon_stddev', str(math.sqrt(acc)))
    gps_bps.set_attribute('noise_seed', '1')
    gps_init_trans = carla.Transform(carla.Location(x= offset, z=2.5))
    gps = world.try_spawn_actor(gps_bps, gps_init_trans, attach_to = vehicle)
    return gps

def calc_err(src, pos):

    return (abs(pos-src)/src) * 100.0


#**********************************************************************************************************************************************************************
















#**************************************************************FUNCTIONS FOR GETTING DISTANCE AND DISTANCE VECTOR*******************************************************


# Getting vector of distance between two vehicles
def get_distance_vec(vehicle1_transform, vehicle2_transform):

    
    vec = vehicle2_transform.location - vehicle1_transform.location

    return np.array([vec.x, vec.y])
   


# FUNCTION FOR GETTING ANGLE BETWEEN TWO VECTORS, WE NEED IT TO DETERMINE RELATIVE LOCATION OF VEHICLES USING EGO VEHICLE AS A REFERENCE
# ANGLE CAN BE CALCULATED USING THE FOLLOWING FORMULA:
#                                                       ANG(RADIANS) = (A1.A2) /  ( ||A1|| . ||A2|| ) 
def get_angle(vec1, vec2):

    res = np.dot(vec1, vec2)
    len1 = np.linalg.norm(vec1)
    len2 = np.linalg.norm(vec2)

    den = len1 * len2

    angle = np.arccos(res/den)
    
    angle = (angle*180)  / np.pi

    return angle





def get_vec_gps(coord_1, coord_2):

    coord_1 = np.array(coord_1)
    coord_2 = np.array(coord_2)

    
    result = coord_2 - coord_1
    result [1] = -1*result[1]

    return result


# FUNCTION FOR GETTING DISTANCE BETWEEN TWO GIVEN LOCATIONS
def get_distance(loc1, loc2):

    return loc1.distance(location = loc2)



# GETTING DISTANCE BETWEEN VEHICLES USING THEIR GPS COORDINATES WHICH ARE PROVIDED USING GNSS SENSOR
def get_dist_gps(coord_v1 , coord_v2 ):

    # converting latitude and longitude to radians
    lat1 = math.radians(coord_v1[1])
    lon1 = math.radians(coord_v1[0])

    lat2 = math.radians(coord_v2[1])
    lon2 = math.radians(coord_v2[0])
 # equation for calculating distance between two coordinates given in radian 
 # the distance returned is in m
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = math.sin(dlat / 2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2)**2
 
    c = 2 * math.asin(math.sqrt(a))
    
    # Radius of earth in kilometers.
    r = 6371
      
    # calculate the distance in meters
    return 1000.0 * (c * r)
    
    # TODO solve problem of actor self destroying


# FUNCTION FOR CONVERTING GPS LONGITUDE AND LATITUDE TO CARTESIAN COORDINATES
def gps_to_cartesian(coord_v):


# TODO: convert gps coordinates to cartesian

    # earth radius in km
    R = 6371

    # convert latitude and longitude to radians
    lat = math.radians(coord_v[1])
    lon = math.radians(coord_v[0])

    # calculating x,y cartesian coordinates
    x = 1000 * R * math.cos(lat) * math.cos(lon)
    y = 1000 * R * math.cos(lat) * math.sin(lon)

    return np.array([x,y])




    
#***********************************************************************************************************************************************************************







#*********************************************************************SCENARIOS***********************************************************************************

# scenario1: traffic light stop
# scenario2: distance using gps
# scenario2: 
# scenario3: 
def scenario_traffic_light():

    # initialize client and world
    client, world = init_client_world(2000)

    # get blueprints and spaawn points from world
    bps, spawn_pts = get_bps_spawn_pts(world)

    #spawning ego vehicle
    vehicle1 = spawn_vehicle(bps, spawn_pts, world)
    
    # spawning camera
    camera = spawn_camera(bps, vehicle1, world,camera_path)
    label = [False]
    model = YOLOV7.load_model()

    camera.listen(lambda img : call_back_camera_yolo(model, img, label))

    # enable auto-pilot for our vehicle
    enable_auto_pilot(client = client, world = world, port = 5000, vehicles_list = [vehicle1] )


    while True:
        # setting spectator to point to the ego vehicle

        spectator = world.get_spectator()
        transform = carla.Transform(vehicle1.get_transform().transform(carla.Location(x=-8, z=2.5)), vehicle1.get_transform().rotation)
        spectator.set_transform(transform)

        if (label[0]):
            stop_vehicle(vehicle1)
        else:
            vehicle1.set_autopilot(True)


        if keyboard.is_pressed('b'):
            camera.stop()
            vehicle1.destroy()
            cv2.destroyAllWindows()
    
def scenario_distance_gps():

    # initialize client and world
    client, world = init_client_world(2000)
    print("hello")
    # get blueprints and spaawn points from world
    bps, spawn_pts = get_bps_spawn_pts(world)

    #spawning ego vehicle
    vehicle1 = spawn_vehicle(bps, spawn_pts, world)
    map = world.get_map()
    waypoint = map.get_waypoint(vehicle1.get_location(), project_to_road = True)
    waypoint = waypoint.next(5)[0]
    vehicle2 = spawn_vehicle_follow(bps, waypoint,vehicle1, world)
    
    vehicle1.set_autopilot(True)

    # spawning gps sensors to our two vehicles
    gps_vehicle1 = spawn_gps_err(bps, vehicle1, world, 1, 5)
    gps_vehicle2 = spawn_gps_err(bps, vehicle2, world, 1, 5)
    gps_vehicle1_2 = spawn_gps_err(bps, vehicle = vehicle1, world = world, offset = 2, acc = 5)

    coord_v1 = [None, None]
    coord_v1_2 = [None, None]
    coord_v2 = [None, None]
  


    gps_vehicle1.listen(lambda gps_coord: store_coord(coord_v1, gps_coord=gps_coord))
    gps_vehicle2.listen(lambda gps_coord: store_coord(coord_v2, gps_coord=gps_coord))
    gps_vehicle1_2.listen(lambda gps_coord: store_coord(coord_v1_2, gps_coord= gps_coord))
    
    print(coord_v1)
    print("\n")
    
    # enable auto-pilot for our vehicles
    # enable_auto_pilot(client = client, world = world, port = 5000, vehicles_list = [vehicle1, vehicle2])


   
    while True:
        # setting spectator to point to the ego vehicle
        
        spectator = world.get_spectator()
        transform = carla.Transform(vehicle1.get_transform().transform(carla.Location(x=-8, z=2.5)), vehicle1.get_transform().rotation)
        spectator.set_transform(transform)


        #FUNCTION FOR PRINTING DISTANCE BETWEEN TWO LOCATIONS
        distance_loc = get_distance(vehicle1.get_transform().location, vehicle2.get_transform().location)
        
        dist_gps = get_dist_gps(coord_v1, coord_v2 )

        distance_vec = get_distance_vec(vehicle1.get_transform(), vehicle2.get_transform())


        vec_forward_gps = get_vec_gps(coord_v1 ,coord_v1_2)
       

        vec_dist_gps = get_vec_gps(coord_v1 , coord_v2)


        print("unit vector of distance vector using location is: ")
        print(carla.Vector2D(x=vehicle1.get_transform().get_forward_vector().x, y=vehicle1.get_transform().get_forward_vector().y).make_unit_vector())

        print("\nunit vetor of distance vector using gps is: ")
        print(carla.Vector2D(x=vec_forward_gps[0], y=vec_forward_gps[1]).make_unit_vector())
        print("\n")
        angle_loc = get_angle([vehicle1.get_transform().get_forward_vector().x,vehicle1.get_transform().get_forward_vector().y], distance_vec)
        angle_gps = get_angle(vec_forward_gps, vec_dist_gps)



        print('The distance using location is: {:.2f} \n '.format(distance_loc) )
        
        print('The distance using gps coodinates is: {:.2f} \n'.format(dist_gps))
        print(f"The angle using location: {angle_loc:.2f}\n")
        print(f"The angle using gps is: {angle_gps:.2f}\n")

        
        print('the error in gps measurment is: {:.2f} \n'.format(calc_err(distance_loc,dist_gps )))
        print("**************************************************************************"+"\n")

        if angle_gps <= 20 and dist_gps <=15:
            stop_vehicle(vehicle1)
            print("Entered Danger Zone.\nStopping ego vehicle.")
            

        if keyboard.is_pressed('b'):
            gps_vehicle1.stop()
            gps_vehicle2.stop()
            gps_vehicle1.destroy()
            gps_vehicle2.destroy()
            vehicle1.destroy()
            vehicle2.destroy()
            cv2.destroyAllWindows()
            
        else:
            continue


# lazem a-call el manual control el awl            <----------------------------------------------------------------------------
def scenario_angle():

    # initialize client and world
    client, world = init_client_world(2000)

    # get blueprints and spaawn points from world
    bps, spawn_pts = get_bps_spawn_pts(world)

    #spawning ego vehicle
    vehicle1 = spawn_vehicle(bps, spawn_pts, world)
    
    # spawning two gps for ego vehicle
    # gps1 = spawn_gps(bps, vehicle1, world)
    # gps2 = spawn_gps_2(bps, vehicle1, world)

    coord_1 = [None, None]
    coord_2 = [None, None]
    # listnening to gps sensors
    # gps1.listen(lambda gps_coord : store_coord(gps_coord, coord_1))
    # gps2.listen(lambda gps_coord : store_coord(gps_coord, coord_2))






    # enable_auto_pilot(client, 5000, [vehicle1], world)

    # TRYING TO GET LOCATION OF A VEHICLE SPAWNED BY ANOTHER CLIENT
    world = client.get_world()
    actors = world.get_actors()
    vehicle2 = actors.filter("vehicle.audi.etron")
    # gps_vehicle2 = actors.filter("sensor.other.gnss")
    vehicle2 = vehicle2[0]
    # spawning gps sensors to our two vehicles
    
    while True:

        distance = get_distance(vehicle1.get_location(), vehicle2.get_location())
        vec1 = vehicle1.get_transform().get_forward_vector()
        vec1 = np.array([vec1.x, vec1.y])
        vec2 = get_distance_vec(vehicle1.get_transform(), vehicle2.get_transform())
        angle = get_angle(vec1, vec2)

        print(f"The distance between vehicles is: {distance:.2f}")
        print("\nThe angle between ego vehicle and other vehicle is: {:.2f} ".format(angle))
        
        

        print("**************************************************************************"+"\n")
        if (distance <=10) and (angle <=20):
            # vehicle1.stop_vehicle()
            print("Stopping ego vehicle")

            break
        if keyboard.is_pressed('b'):
            vehicle1.destroy()
            cv2.destroyAllWindows()
            
            break
        else:
            continue

def scenario_kalman():
    client , world = init_client_world(2000)
    bps, spawn_pts = get_bps_spawn_pts(world)
    vehicle = spawn_vehicle(bps, spawn_pts, world)

    

    camera_bps = bps.find("sensor.camera.rgb")
    camera_trans = carla.Transform(carla.Location(x = 10, y = 0, z= 10), carla.Rotation(pitch = -10.0))
    camera = world.spawn_actor(camera_bps, camera_trans, attach_to= vehicle)
    out_img = np.zeros((1280,720))
    camera.listen(lambda img : call_back_camera(img,out_img,  True))

    

    vehicle.set_autopilot(True)
    spectator = world.get_spectator()
    transform = carla.Transform(vehicle.get_transform().transform(carla.Location(x=-8, z=2.5)), vehicle.get_transform().rotation)
    spectator.set_transform(transform)
    while True:
        if keyboard.is_pressed == 'b':
            camera.stop()
            break
            
        else: 
            continue
    cv2.destroyAllWindows()


def spawn_two_op_vehicles():
    client,world = init_client_world(2000)

    bps, spawn_pts = get_bps_spawn_pts(world)

    vehicle1 = spawn_vehicle(bps, spawn_pts, world)
    map = world.get_map()
    waypoint = map.get_waypoint(vehicle1.get_location(), project_to_road = True)
    waypoint = waypoint.next(5)[0]
    vehicle2 = spawn_vehicle_follow(bps, waypoint,vehicle1, world)

    enable_auto_pilot(client, 5000, [vehicle1], world)
    spectator = world.get_spectator()
    spectator_trans = carla.Transform(vehicle1.get_transform().transform(carla.Location(x=-8, z= 2.5)), vehicle1.get_transform().rotation)
    spectator.set_transform(spectator_trans)
    while True:

        

        if keyboard.is_pressed('b'):
            vehicle1.destroy()
            vehicle2.destroy()
            break

def lane_detection_output(image, src_pts, dst_pts,s_thresh, l_thresh,shadow_thresh,right_poly_old,left_poly_old, size,  debug = 0):

    output_name = ''
    debug_resize = 3
    ny_pipeline = 3
    if debug == 1:
        pipeline = []
        output_name += '_debug'
        size_debug = ((size[0] // debug_resize) * 2 , (size[1] // debug_resize) * ny_pipeline)
    

    # if right_poly_old is None or left_poly_old is None:
    #     print("ana dkhlt")
    #     right_poly_old = np.zeros((size[1] , 2) , np.int32)
    #     left_poly_old =  np.zeros((size[1] , 2) , np.int32)

    

    frame = image
    
        
    (combined_binary,l,s) = detect_edges(frame , s_thresh , l_thresh,shadow_thresh)
#         kernel_op = np.asarray([[0 , 1, 0],
#                                [0, 1 , 0],
#                                [0 , 1, 0]] , np.uint8)
    kernel_co = np.ones((3,3) , np.uint8)
#         combined_binary = cv2.morphologyEx(combined_binary, cv2.MORPH_OPEN ,kernel_op)
    combined_binary = cv2.dilate(combined_binary,kernel_co)
    M,Minv = PerspectiveTransform(src_pts, dst_pts)
    dst = warpPerspective(combined_binary ,M , size)
    dst_equalized = cv2.equalizeHist(dst)     
#         dst_colored = perspective_warp(frame ,src=input_points , dst=p2)
    dst_colored = warpPerspective(frame ,M , size)
    # print(f"right poly old before call back is: {right_poly_old}\n")
    out_img,right_poly_new,left_poly ,left_fit , right_fit,ploty= sliding_window(dst , dst_colored, (30 , 40),right_poly_old,left_poly_old,80)
    right_poly_old = right_fit
    left_poly_old = left_fit
    
    left = np.array([np.transpose(np.vstack([left_fit, ploty]))])
    right = np.array([np.flipud(np.transpose(np.vstack([right_fit, ploty])))])
    points = np.hstack((left, right))
    
    
    
    cv2.fillPoly(out_img,np.int_(points),color= (255,0,0))
    
    re_bird = warpPerspective(out_img , Minv , size) 
    
    
    #printing the off-centre distance on video frame
    # dst_off = calc_off_dist(M,frame ,right_poly_new[200], left_poly[200])
    # re_bird = cv2.putText(img=re_bird, text='The car is '+str(dst_off)+' off centre' , org=(0,150), fontFace=cv2.FONT_HERSHEY_TRIPLEX, fontScale=1, color=(255, 255, 255),thickness=3)
    re_bird = cv2.putText(img=re_bird, text='The car is off centre' , org=(0,150), fontFace=cv2.FONT_HERSHEY_TRIPLEX, fontScale=1, color=(255, 255, 255),thickness=3)
    image = np.zeros_like(re_bird)
    cv2.addWeighted(frame, 0.5, re_bird, 0.5,0, image)
    
    if debug == 1:
        pipeline.append(l)
        pipeline.append(s)
        pipeline.append(combined_binary)
        pipeline.append(dst)
        pipeline.append(dst_equalized)
        pipeline.append(image)
        image = debug_mode(pipeline , debug_resize)
        pipeline.clear()
        
        
    return image, right_poly_old, left_poly_old
            
       
def scenario_lane_detection():


    client, world = init_client_world(2000)
    size = (800, 600)
    sync_world(world)

    bps, spawn_pts = get_bps_spawn_pts(world)
    
    vehicle = spawn_vehicle(bps,spawn_pts, world)
    enable_auto_pilot(client, 5000, [vehicle], world)

    camera = spawn_camera(bps, vehicle, world, camera_path)
    camera_img = [None]
    # configuring parameters for lane detection algorrithm:
    # points for prespective transform
    # input_top_left = [550,468]
    # input_top_right = [742,468]
    # input_bottom_right = [1280,720]
    # input_bottom_left = [128,720]
    input_top_left = [320,350]
    input_top_right = [460,350]
    input_bottom_right = [750,600]
    input_bottom_left = [60,600]

    s_thresh = (75, 255)
    l_thresh = (140 , 255)
    shad_thresh = (150,100)

    src_pts = np.float32([input_bottom_left,input_top_left,input_top_right,input_bottom_right])
    dst_pts = np.float32([[0,size[1]],[0,0],[size[0],0],[size[0],size[1]]])
    
    right_poly_old, left_poly_old = np.zeros((600,1), np.int32), np.zeros((600,1), np.int32)
    old_list = [right_poly_old, left_poly_old]

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_path = '/home/amr/GradProj/videos/lane_demo.mp4'
    out_src = cv2.VideoWriter(video_path, fourcc, 25.0, (800,600))
    video_path = '/home/amr/GradProj/videos/lane_sliding_window.mp4'
    out_detect = cv2.VideoWriter(video_path, fourcc, 25.0, (800,600))
    
    # camera.listen(lambda img : call_back_camera_lane(img, camera_img, src_pts, dst_pts, s_thresh, l_thresh, shad_thresh, old_list, out_src,out_detect, True) )
    camera.listen(lambda img : call_back_camera(img, camera_img, True))
    
    
    

   

    
    while(True):

        world.tick()
        spectator = world.get_spectator()
        spectator_transform = carla.Transform(vehicle.get_transform().transform(carla.Location(x=-8, z= 2.5)), vehicle.get_transform().rotation)
        spectator.set_transform(spectator_transform)
        
        
        # cv2.imshow("Lane Detection: ", camera_img)

        # if cv2.waitKey(1) & 0xFF == ord('b'):
        #     break
        try:

            if keyboard.is_pressed('b'):
                out_src.release()
                out_detect.release()
                camera.stop()
                camera.destroy()
                vehicle.destroy()
                desync_world(world)
                break
                
        except:
            continue
    
            
        

def extract_color_trial():
    img = cv2.imread("/home/amr/GradProj/output/22051.png")
    out_img = Phase1.extract_white(img)
    cv2.imshow("img", out_img)
    if cv2.waitKey(0) & 0xFF == ord("q"):
        cv2.destroyAllWindows()

if __name__ == '__main__':
    scenario_distance_gps()




