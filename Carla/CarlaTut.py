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
    vehicle = world.try_spawn_actor(vehicle_bp, random.choice(spawn_pts))
    return vehicle






def spawn_vehicle_follow(blue_prints, vehicle, world):
     
     return world.try_spawn_actor(blue_prints.find("vehicle.audi.tt"), transform =vehicle.get_transform().transform(carla.Location(x=5, y= 0, z= 0)) )





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
    
    tm = client.get_trafficmanager()
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
#     init_settings = world.get_settings()
#     settings = world.get_settings()
#     settings.synchronous_mode = True
    
    # After that, set the TM to sync mode
#     tm.set_synchronous_mode(True)


    # Tick the world in the same client
#     world.apply_settings(init_settings)
#     world.tick()



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

def call_back_camera(img):
    init_shape = (img.height, img.width, 4)
    image = np.array(img.raw_data).reshape(init_shape)[:,:,:3]

    cv2.imshow('img',image)
    _ = cv2.waitKey(1)
        

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


def spawn_gps(blue_print, vehicle, world):
    gps_bp = blue_print.find('sensor.other.gnss')
    gps_init_trans = carla.Transform(carla.Location(x= 1, z=2.5))
    gps = world.try_spawn_actor(gps_bp, gps_init_trans, attach_to = vehicle)
    
    # gps.listen(lambda gps_coord: store_coord(latitude=latitude, longitude= longitude, gps_coord=gps_coord))

    return gps

def spawn_gps_2(blue_print, vehicle, world):

    gps_bp = blue_print.find('sensor.other.gnss')
    gps_transform = carla.Transform(carla.Location(x = 2, z = 2.5))
    gps = world.spawn_actor(gps_bp, gps_transform, attach_to = vehicle)

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
    vehicle2 = spawn_vehicle(bps, spawn_pts, world)
    

    # spawning gps sensors to our two vehicles
    gps_vehicle1 = spawn_gps(bps, vehicle1, world)
    gps_vehicle2 = spawn_gps(bps, vehicle2, world)
    gps_vehicle1_2 = spawn_gps_2(bps, vehicle = vehicle1, world = world)

    coord_v1 = [None, None]
    coord_v1_2 = [None, None]
    coord_v2 = [None, None]
  


    gps_vehicle1.listen(lambda gps_coord: store_coord(coord_v1, gps_coord=gps_coord))
    gps_vehicle2.listen(lambda gps_coord: store_coord(coord_v2, gps_coord=gps_coord))
    gps_vehicle1_2.listen(lambda gps_coord: store_coord(coord_v1_2, gps_coord= gps_coord))
    
    print(coord_v1)
    print("\n")
    
    # enable auto-pilot for our vehicles
    enable_auto_pilot(client = client, world = world, port = 5000, vehicles_list = [vehicle1, vehicle2])


   
    while True:
        # setting spectator to point to the ego vehicle
        
        spectator = world.get_spectator()
        transform = carla.Transform(vehicle1.get_transform().transform(carla.Location(x=-8, z=2.5)), vehicle1.get_transform().rotation)
        spectator.set_transform(transform)

        print("\n")
        print(vehicle1.get_transform().location)
        print("\n")


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


        if keyboard.is_pressed('b'):
            gps_vehicle1.stop()
            gps_vehicle2.stop()
            vehicle1.destroy()
            vehicle2.destroy()
            cv2.destroyAllWindows()
            break
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
        if (distance <=5) and (angle <=20):
            # vehicle1.stop_vehicle()
            print("Stopping ego vehicle")
        if keyboard.is_pressed('b'):
            vehicle1.destroy()
            cv2.destroyAllWindows()
            
            break
        else:
            continue




scenario_angle()




