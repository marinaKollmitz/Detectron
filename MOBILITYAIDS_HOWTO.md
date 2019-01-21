# Detection of People with Mobility Aids

Here, we describe how use the DetectronDistance repository for detecting people and distinguishing them according to the mobility aids they use.

In the following, We'll call the directory that you cloned DetectronDepth into `$DETECTRON_ROOT` and the folder where you keep your datasets `$DATASETS_DIR`. 

## Run the people detection ROS node

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

## Get Trained Mobility Aids Models

Download and unpack the trained mobilityaids models we used in our RAS publication.

```
cd $DETECTRON_ROOT
wget http://mobility-aids.informatik.uni-freiburg.de/mobilityaids_models.zip
unzip mobilityaids_models.zip
```

## Testing

To test the detection performance of our models on the mobilityaids dataset, use the `tools/test_net.py` script for the model you want to test. For example, to test the VGG-M RGB model on the mobilityaids test set, run 
```
cd $DETECTRON_ROOT 
python2 tools/test_net.py --cfg mobilityaids_models/VGG-M/faster_rcnn_VGG-M_RGB.yaml TEST.WEIGHTS mobilityaids_models/VGG-M/train/mobilityaids_RGB_train/model_final.pkl 
```
The script will output image APs and depth error APs at 0.25 and 0.5 meters. It will further save various results files to the output directory, some of which are used by the tracking module. The script uses our mobilityaids matlab evaluation. If you do not have matlab installed, you can use the coco evaluation to verify that the image detection is working
```
cd $DETECTRON_ROOT 
# to evaluate with coco if you dont have matlab
python2 tools/test_net.py --cfg mobilityaids_models/VGG-M/faster_rcnn_VGG-M_RGB.yaml TEST.WEIGHTS mobilityaids_models/VGG-M/train/mobilityaids_RGB_train/model_final.pkl TEST.FORCE_JSON_DATASET_EVAL True
```
but it will not produce the same results (because it uses interpolated AP) and also generate no files.
