# Detection of People with Mobility Aids

Here, we describe how use the DetectronDistance repository for detecting people and distinguishing them according to the mobility aids they use.

In the following, We'll call the directory that you cloned DetectronDepth into `$DETECTRON_ROOT` and the folder where you keep your datasets `$DATASETS_DIR`. 

## Run the people detection ROS node

If you only want to use the mobilityaids people detector as a ROS node, you don't need to do any of the following. Check out the [mobilityaids_detector repository](https://github.com/marinaKollmitz/mobilityaids_detector) and follow the instructions there.

## Download and Prepare the Dataset

Download the dataset into a `$DATASETS_DIR` of your choice

```
# setup folder for dataset
cd $DATASETS_FOLDER
mkdir mobility-aids && cd mobility-aids
# download zipped dataset files
wget -i $DETECTRON_ROOT/detectron/datasets/data/download_mobility_aids.txt
# unzip files
unzip \*.zip
# create link
ln -sv $DATASETS_DIR/mobility-aids/ $DETECTRON_ROOT/detectron/datasets/data/
```
