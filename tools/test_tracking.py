import json
import numpy as np
from multiclass_tracking.tracker import Tracker
from multiclass_tracking.viz import Visualizer
from multiclass_tracking.image_projection import ImageProjection
import tf
from detectron.datasets.json_dataset import JsonDataset

import argparse
import sys
import os

def parse_args():
    parser = argparse.ArgumentParser(description='Test the multiclass tracking modul on detections')
    
    parser.add_argument(
        '--validation-dir',
        dest='validation_dir',
        help='validation directory',
        default=None
    )
    parser.add_argument(
        '--step', 
        dest='step', 
        help='(optional) wait for key stroke after each observation', 
        action='store_true'
    )
    parser.add_argument(
        '--visualize',
        dest='viz',
        help='(optional) visualize tracking with matplotlib',
        action='store_true'
    )
    parser.add_argument(
        '--datasets',
        nargs='+', #allow multiple arguments
        dest='datasets',
        help='dataset(s) to evaluate',
        default=None
    )
    parser.add_argument(
        '--detections',
        nargs='+', #allow multiple arguments
        dest='detection_files',
        help='json file(s) with detections for dataset(s), generate with test_net.py',
        default=None
    )
    parser.add_argument(
        '--ekf-only',
        dest='ekf_only',
        help='(optional) only evaluate ekf without hmm',
        action='store_true'
    )
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    
    args = parser.parse_args()
    
    #TODO mandatory args
    return args

def load_detections(detection_file, dataset, thresholds):
    
    detections = json.load(open(detection_file))
    roidb = dataset.get_roidb()
    classes = dataset.classes
    
    detections_per_image = []

    for entry in roidb:
        #get all detections for image
        im_dets = [det for det in detections if det['image_id'] == entry['id']]
        
        #apply threshold
        im_dets_thresh = [det for det in im_dets if det['score'] > thresholds[classes[det['category_id']]]]
        
        detections_per_image.append(im_dets_thresh)
    
    return detections_per_image

def load_class_thresholds(thresholds_file, dataset):
    
    thresholds = {}
    default_thresh = 0.9
    
    if thresholds_file is not None:
    
        with open(thresholds_file) as f:
            for line in f:
                (key, val) = line.split(',')
                thresholds[key] = float(val)
                if key not in dataset.classes:
                    print "warn: threshold file has unknown entry: ", key
    else:
        
        print "setting default class threshold of", default_thresh
        
        for cla in dataset.classes:
            if cla != '__background__':
                thresholds[cla] = default_thresh

    return thresholds

def get_transform_matrix(trafo_dict):
    
    trans = trafo_dict['translation']
    rot = trafo_dict['rotation']
    trafo = tf.transformations.quaternion_matrix([rot['x'], rot['y'], rot['z'], rot['w']])
    trafo[0:3,3] = [trans['x'], trans['y'], trans['z']]
    
    return trafo

def do_kalman_filtering(im_detections, dataset, time_delta, ekf_sensor_noise, 
                        hmm_observation_model, viz=False, step=False, use_hmm=True):
    
    tracked_im_dets = []
    
    num_timesteps = len(im_detections)
    
    #get information from the datasets
    roidb = dataset.get_roidb()
    cam_calib = roidb[0]['camera_calibration']
    
    trafo_cam_in_robot = get_transform_matrix(roidb[0]['base_cam_trafo'])
    
    #initialize tracker
    tracker = Tracker(cam_calib, ekf_sensor_noise, hmm_observation_model, use_hmm)
    
    #visualization module
    if viz:
        visualizer = Visualizer(len(dataset.classes))
    
    for timestep in range(num_timesteps):
        
        #get transformation from odom frame in camera frame
        trafo_robot_in_odom = get_transform_matrix(roidb[timestep]['odom'])
        trafo_cam_in_odom = np.dot(trafo_robot_in_odom, trafo_cam_in_robot)
        trafo_odom_in_cam = np.linalg.inv(trafo_cam_in_odom)
        
        #get detections from this timestep
        detections_timestep = im_detections[timestep]
        
        #tracker prediction step
        tracker.predict(time_delta)
        
        #tracker update step
        tracker.update(detections_timestep, trafo_odom_in_cam)
        
        #save tracking detection
        for track in tracker.tracks:
            cla = track.get_class()
                
            odom_det = {}
            odom_det["x"] = track.mu[0,0]
            odom_det["y"] = track.mu[1,0]
            odom_det["z"] = track.mu[2,0]
            
            bbox_width = track.bbox[2]
            bbox_height = track.bbox[3]
            
            cam_det = ImageProjection.transform_detection(odom_det, trafo_odom_in_cam)
            im_bbox = ImageProjection.get_image_bbox(cam_det, cam_calib, bbox_width, bbox_height)
            
            #check if bbox center is inside the image
            im_x = im_bbox[0] + im_bbox[2]/2
            
            if im_x > 0 and im_x < 960:
                tracking_det = {}
                tracking_det["category_id"] = cla
                tracking_det["bbox"] = im_bbox
                tracking_det["depth"] = cam_det["z"]
                tracking_det["score"] = track.get_score()
                tracking_det["image_id"] = roidb[timestep]['id']
                
                tracked_im_dets.append(tracking_det)
        
        if viz:
            visualizer.visualize_detections(roidb[timestep]['image'], 
                                            trafo_cam_in_odom, 
                                            trafo_cam_in_robot,
                                            cam_calib,
                                            detections_timestep, 
                                            tracker.tracks, 
                                            time_delta=0.06, 
                                            step=step)
            
    return tracked_im_dets
    
if __name__ == '__main__':
    args = parse_args()
    all_dets = []
    
    for i in range(len(args.datasets)):
        dataset = JsonDataset(args.datasets[i])
    
        #load class thresholds 
        class_thresh_file = os.path.join(args.validation_dir + 'AP_thresholds.txt')
        cla_thresh = load_class_thresholds(class_thresh_file, dataset)
        
        #load hmm observation model
        hmm_model_file = os.path.join(args.validation_dir + 'observation_model.txt')
        hmm_observation_model = np.loadtxt(hmm_model_file, delimiter=',')
        
        #load ekf sensor noise
        ekf_sensor_noise_file = os.path.join(args.validation_dir + 'meas_cov.txt')
        ekf_sensor_noise = np.loadtxt(ekf_sensor_noise_file, delimiter=',')
        
        #load detection per image
        detections = load_detections(args.detection_files[i], dataset, cla_thresh)
        
        #visualization options
        step = args.step
        viz = args.viz
        
        #use hmm
        use_hmm = not args.ekf_only
        
        #do the filtering
        tracked_im_dets = do_kalman_filtering(detections, dataset, 0.06, ekf_sensor_noise, 
                                              hmm_observation_model, use_hmm=use_hmm, 
                                              viz=args.viz, step=args.step)
        
        #write filtered detections to file
        if use_hmm:
            detections_out = os.path.splitext(args.detection_files[i])[0] + "_EKF_HMM.json"
            
        else:
            detections_out = os.path.splitext(args.detections_file[i])[0] + "_EKF.json"
            
        with open(detections_out,'w') as out_file:
            json.dump(tracked_im_dets, out_file)
            
            print "wrote filtered detections to: ", detections_out