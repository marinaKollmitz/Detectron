# Installing DetectronDistance

This document explains how to install our adapted DetectronDistance, its dependencies (including Caffe2), and the COCO dataset. See [Install.md](INSTALL.md) for the original Detectron instructions.
- For general information about Detectron, please see [`README.md`](README.md).

**Requirements:**

- NVIDIA GPU, Linux, Python2
- Caffe2, various standard Python packages, and the COCO API; Instructions for installing these dependencies are found below

**Notes:**

- Detectron operators currently do not have CPU implementation; a GPU system is required.
- Detectron has been tested extensively with CUDA 8.0 and cuDNN 6.0.21.

## Caffe2

To install Caffe2 with CUDA support, follow the [installation instructions](https://caffe2.ai/docs/getting-started.html) from the [Caffe2 website](https://caffe2.ai/). **If you already have Caffe2 installed, make sure to update your Caffe2 to a version that includes the [Detectron module](https://github.com/caffe2/caffe2/tree/master/modules/detectron).**

Please ensure that your Caffe2 installation was successful before proceeding by running the following commands and checking their output as directed in the comments.

```
# To check if Caffe2 build was successful
python2 -c 'from caffe2.python import core' 2>/dev/null && echo "Success" || echo "Failure"

# To check if Caffe2 GPU build was successful
# This must print a number > 0 in order to use Detectron
python2 -c 'from caffe2.python import workspace; print(workspace.NumCudaDevices())'
```

If the `caffe2` Python package is not found, you likely need to adjust your `PYTHONPATH` environment variable to include its location (`/path/to/caffe2/build`, where `build` is the Caffe2 CMake build directory).

## Other Dependencies

Install the [COCO API](https://github.com/cocodataset/cocoapi):

```
# COCOAPI=/path/to/clone/cocoapi
git clone https://github.com/cocodataset/cocoapi.git $COCOAPI
cd $COCOAPI/PythonAPI
# Install into global site-packages
make install
# Alternatively, if you do not have permissions or prefer
# not to install the COCO API into global site-packages
python2 setup.py install --user
```

Note that instructions like `# COCOAPI=/path/to/install/cocoapi` indicate that you should pick a path where you'd like to have the software cloned and then set an environment variable (`COCOAPI` in this case) accordingly.

## DetectronDistance

Clone the DetectronDistance repository:

```
# DETECTRON_ROOT=/path/to/clone/DetectronDistance
git clone https://github.com/marinaKollmitz/DetectronDistance.git $DETECTRON_ROOT
```

Install Python dependencies:

```
pip install -r $DETECTRON_ROOT/requirements.txt
```

Set up Python modules:

```
cd $DETECTRON_ROOT && make
```

Check that Detectron tests pass (e.g. for [`SpatialNarrowAsOp test`](tests/test_spatial_narrow_as_op.py)):

```
python2 $DETECTRON_ROOT/detectron/tests/test_spatial_narrow_as_op.py
```
