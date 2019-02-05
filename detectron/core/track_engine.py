from multiclass_tracking.tracker import Tracker
from multiclass_tracking.viz import Visualizer
from multiclass_tracking.image_projection import ImageProjection

from detectron.datasets.json_dataset import JsonDataset
from detectron.datasets.task_evaluation import evaluate_tracking
from detectron.datasets.json_dataset_evaluator import _write_coco_bbox_results_file
from detectron.core.test_engine import test_net, test_net_on_dataset
from detectron.core.config import get_output_dir

import os
import tf
import json
import numpy as np
import logging

logger = logging.getLogger(__name__)

def get_detections(weights_file, dataset, class_thresh):
    
    output_dir = os.path.abspath(get_output_dir(dataset.name, training=False))
    
    res_file = os.path.join(
        output_dir, 'bbox_' + dataset.name + '_results.json'
    )
    
    #generate detections if they do not exist
    if not os.path.exists(res_file):
        
        logger.info('results file %s does not exist. Generating results '
                    'for dataset: %s' % (res_file, dataset.name))
        
        all_boxes, all_depths, all_segms, all_keyps = test_net(
                weights_file, dataset.name, None, output_dir)
        
        _write_coco_bbox_results_file(dataset, all_boxes, all_depths, res_file)
        
    detections = load_detections(res_file, dataset, class_thresh)
    
    return detections

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
    
    roidb = dataset.get_roidb()
    num_images = len(roidb)
    num_classes = len(dataset.classes)

    filtered_boxes  = [[np.empty([0,5]) for _ in range(num_images)] for _ in range(num_classes)]
    filtered_depths = [[np.empty([0,1]) for _ in range(num_images)] for _ in range(num_classes)]
    
    #get information from the datasets
    cam_calib = roidb[0]['camera_calibration']
    
    trafo_cam_in_robot = get_transform_matrix(roidb[0]['base_cam_trafo'])
    
    #initialize tracker
    tracker = Tracker(cam_calib, ekf_sensor_noise, hmm_observation_model, use_hmm)
    
    #visualization module
    if viz:
        visualizer = Visualizer(num_classes)
    
    for timestep in range(num_images):
        
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
                
                bbox_xyxy = [im_bbox[0], im_bbox[1], im_bbox[0]+im_bbox[2], im_bbox[1]+im_bbox[3]]
                
                #append filtered detection
                filtered_boxes[cla][timestep] = np.append(filtered_boxes[cla][timestep], 
                              [np.append(bbox_xyxy, track.get_score())], axis=0)
                filtered_depths[cla][timestep] = np.append(filtered_depths[cla][timestep], 
                               cam_det["z"])
        
        if viz:
            visualizer.visualize_detections(roidb[timestep]['image'], 
                                            trafo_cam_in_odom, 
                                            trafo_cam_in_robot,
                                            cam_calib,
                                            detections_timestep, 
                                            tracker.tracks, 
                                            time_delta=0.06, 
                                            step=step)
    
    return filtered_boxes, filtered_depths

def write_filtered_detections(dataset, filtered_boxes, filtered_depths, use_hmm):
    
    output_dir = os.path.abspath(get_output_dir(dataset.name, training=False))
    
    if use_hmm:
        res_file = os.path.join(
                output_dir, 'bbox_' + dataset.name + '_results_EKF_HMM.json'
        )
        
    else:
        res_file = os.path.join(
                output_dir, 'bbox_' + dataset.name + '_results_EKF.json'
        )
        
    _write_coco_bbox_results_file(dataset, filtered_boxes, filtered_depths, res_file)
    
    return res_file
    
# Only the detectron.datasets.mobilityaids_dataset_evaluator writes the 
# necessary validation results so far. It is automatically called if 
# the dataset name includes 'mobilityaids'. You can force it by setting 
# cfg.FORCE_MOBILITYAIDS_EVAL to true. the mobilityaids_dataset_evaluator
# requires matlab.
def validate_tracking_params(weights_file,
                             dataset):
    
    output_dir = os.path.abspath(get_output_dir(dataset.name, training=False))
    
    #tracking parameter files
    class_thresh_file = os.path.join(output_dir, 'AP_thresholds.txt')
    obs_model_file = os.path.join(output_dir, 'observation_model.txt')
    meas_cov_file = os.path.join(output_dir, 'meas_cov.txt')
    
    #generate param files if they do not exist
    if (not os.path.exists(class_thresh_file) or 
        not os.path.exists(obs_model_file) or 
        not os.path.exists(meas_cov_file)):
            logger.info('validation files for validation dataset %s do not '
                    'exist. Generating them now' % (dataset.name))
            
            #this performs inference on all images and calls the evaluation 
            #script, generating the validation files
            test_net_on_dataset(weights_file,dataset.name,None,output_dir)
            
    #read parameters from files
    cla_thresh = load_class_thresholds(class_thresh_file, dataset)
    observation_model = np.loadtxt(obs_model_file, delimiter=',')
    ekf_sensor_noise = np.loadtxt(meas_cov_file, delimiter=',')
    
    return cla_thresh, observation_model, ekf_sensor_noise

def run_tracking(validation_dataset, tracking_datasets, weights_file, timestep,
                 use_hmm=True, visualize=False, step=False):
    validation_set_json = JsonDataset(validation_dataset)
    class_thresh, obs_model, meas_cov  = validate_tracking_params(weights_file, 
                                                                  validation_set_json)
    
    res_files = []
    json_datasets = []
    
    for i in range(len(tracking_datasets)):
        
        tracking_set_name = tracking_datasets[i]
        tracking_set_json = JsonDataset(tracking_set_name)
        
        detections = get_detections(weights_file,
                                    tracking_set_json,
                                    class_thresh)
        
        filtered_boxes, filtered_depths = do_kalman_filtering(detections, 
                    tracking_set_json, timestep, meas_cov, obs_model, 
                    use_hmm=use_hmm, viz=visualize, step=step)
        
        res_file = write_filtered_detections(tracking_set_json, 
                                             filtered_boxes, filtered_depths, 
                                             use_hmm)
        
        res_files.append(res_file)
        json_datasets.append(tracking_set_json)
        
    #perform the evaluation for all datasets
    tracking_dataset_name = '-'.join(tracking_datasets)
    tracking_output_dir = os.path.abspath(get_output_dir(tracking_dataset_name, training=False))
    evaluate_tracking(res_files, json_datasets, tracking_output_dir, use_matlab=True)