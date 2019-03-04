# Detection of People with Mobility Aids

Here, we describe how use the DetectronDistance repository for detecting people and distinguishing them according to the mobility aids they use.

In the following, We'll call the directory that you cloned DetectronDepth into `$DETECTRON_ROOT` and the folder where you keep your datasets `$DATASETS_DIR`. 

## Installation

Please follow the [Installation Instructions](INSTALL_DETECTRONDISTANCE.md) for the DetectronDistance code.

## Get Trained Mobility Aids Models

Download and unpack the trained mobilityaids models we used in our RAS publication.

```
cd $DETECTRON_ROOT
wget http://mobility-aids.informatik.uni-freiburg.de/mobilityaids_models.zip
unzip mobilityaids_models.zip
```

## Run the mobilityaids detection ROS node

If you only want to use the mobilityaids people detector as a ROS node, you don't need to do any of the following. Check out the [mobilityaids_detector repository](https://github.com/marinaKollmitz/mobilityaids_detector) and follow the instructions there.

## Download and Prepare the Dataset

Download and unpack the dataset into a `$DATASETS_DIR` of your choice
```
# setup folder for dataset
cd $DATASETS_FOLDER
mkdir mobility-aids && cd mobility-aids
# download zipped dataset files
wget -i $DETECTRON_ROOT/detectron/datasets/mobility_aids/download_mobility_aids.txt
# unzip files
unzip \*.zip
# create link
ln -sv $DATASETS_DIR/mobility-aids/ $DETECTRON_ROOT/detectron/datasets/data/
```

Now generate coco-format labels for the mobilityaids dataset
```
python2 $DETECTRON_ROOT/detectron/datasets/mobility_aids/generate_mobilityaids_coco_labels.py
```

## Testing

### Test Detection

To test the detection performance of our models on the mobilityaids dataset, use the `tools/test_net.py` script for the model you want to test. For example, to test the VGG-M RGB model on the mobilityaids test set, run 
```
cd $DETECTRON_ROOT 
python2 tools/test_net.py --cfg mobilityaids_models/VGG-M/faster_rcnn_VGG-M_RGB.yaml
```
The script will output image APs and depth error APs at 0.25 and 0.5 meters. It will further save various results files to the output directory, some of which are used by the tracking module. The script uses our mobilityaids matlab evaluation. If you do not have matlab installed, you can use the coco evaluation to verify that the image detection is working
```
cd $DETECTRON_ROOT 
# to evaluate with coco if you dont have matlab
python2 tools/test_net.py --cfg mobilityaids_models/VGG-M/faster_rcnn_VGG-M_RGB.yaml TEST.FORCE_JSON_DATASET_EVAL True
```
but it will not produce the same results (because it uses interpolated AP) and also generates no files.

### Test Tracking

For tracking you first need the [multiclass-people-tracking](https://github.com/marinaKollmitz/multiclass-people-tracking) code. Clone it into a directory of your choice which we will refer to as `$TRACKING_ROOT`:
```
cd $TRACKING_ROOT
git clone https://github.com/marinaKollmitz/multiclass-people-tracking
```
To make sure python can find the tracking code you can add it to your `$PYTHONPATH` by adding the following to your `.bashrc`:
```
export PYTHONPATH=$PYTHONPATH:$TRACKING_ROOT/multiclass-people-tracking/
```
To test the performance of our probabilistic position, velocity and class estimation module, use the `tools/test_tracking.py` script for the model you want to test. For example, to test the tracking performance for the VGG-M RGB model on the mobilityaids tracking datasets, run
```
cd $DETECTRON_ROOT 
python2 tools/test_tracking.py --cfg mobilityaids_models/VGG-M/faster_rcnn_VGG-M_RGB.yaml
```
To evaluate the tracking results you need matlab. If you just want to look at the tracking in action, you can use the `--visualize` option to visualize the detections before and after filtering (does not require matlab). Use the `--step` option to pause between frames. Press any key to go to the next frame. Use the `--ekf-only` option if you want to test the performance without the HMM mobule and the `--no-filtering` option if you just want to evaluate precision and recall for the thresholded detections, without filtering.

## Training

### get pretrained models

### with InOutDoor examples
For the RAS paper, we trained our mobilityaids models with additional examples from the InOutDoor dataset, seq. 0-2. To this end, we enhanced the InOutDoor annotations with centroid depth labels. 

Download and unpack the enhanced InOutDoor labels:
```
# go to mobilityaids dataset folder
cd $DATASETS_FOLDER/mobility-aids/ 
# download zipped InOutDoor annotations
wget http://mobility-aids.informatik.uni-freiburg.de/dataset/Annotations_InOutDoor_DepthJet.zip
wget http://mobility-aids.informatik.uni-freiburg.de/dataset/Annotations_InOutDoor_RGB.zip
# unzip annotations
unzip Annotations_InOutDoor_DepthJet.zip 
unzip Annotations_InOutDoor_RGB.zip
```
Download and unpack the InOutDoor dataset:
```
# go to dataset folder
cd $DATASET_FOLDER
# download InOutDoor dataset
wget http://adaptivefusion.cs.uni-freiburg.de/dataset/InOutDoorPeopleRGBD.zip
# unpack dataset
unzip InOutDoorPeopleRGBD.zip 'InOutDoorPeopleRGBD/DepthJetQhd.tar.gz' 'InOutDoorPeopleRGBD/ImagesQhd.tar.gz'
# unpack Image folders
cd InOutDoorPeopleRGBD
tar -zxvf ImagesQhd.tar.gz 
tar -zxvf DepthJetQhd.tar.gz 
```
Now you need to create links from the mobilityaids image folders to the InOutDoor images
```
ln -sv $DATASETS_DIR/InOutDoorPeopleRGBD/ImagesQhd/* $DATASETS_DIR/mobility-aids/Images_RGB/
ln -sv $DATASETS_DIR/InOutDoorPeopleRGBD/DepthJetQhd/* $DATASETS_DIR/mobility-aids/Images_DepthJet/
```
Finally, to generate the new mobility aids label files with InOutDoor examples, run
```
python2 $DETECTRON_ROOT/detectron/datasets/mobility_aids/generate_mobilityaids_coco_labels.py --with_InOutDoor
```
To train a model, e.g. the VGG-M model on RGB data, run the `train_net.py` script:
```
cd $DETECTRON_ROOT
python2 tools/train_net.py --cfg mobilityaids_models/VGG-M/faster_rcnn_VGG-M_RGB.yaml 
```
### without InOutDoor examples

If you want to train a DetectronDistance model without the additional InOutDoor examples, specify the `mobilityaids_<DepthJet/RGB>_train` dataset for training. You can do this by changing the `TRAIN.DATASETS` entry in the `.yaml` config file. For example, for training a VGG-M network on RGB data without InOutDoor examples, open `$DETECTRON_ROOT/mobilityaids_models/VGG-M/faster_rcnn_VGG-M_RGB.yaml` and change it to:
```
...
TRAIN:
   ...
   DATASETS: ('mobilityaids_RGB_train,)
   ...
```
To train, run the `train_net.py` script with your adapted `.yaml` config:
```
cd $DETECTRON_ROOT
python2 tools/train_net.py --cfg mobilityaids_models/VGG-M/faster_rcnn_VGG-M_RGB.yaml 
```
The test results will be slightly worse that training with the InOutDoor examples, especially for the pedestrian class.
