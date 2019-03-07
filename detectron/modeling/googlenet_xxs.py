# Copyright (c) 2017-present, Facebook, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
##############################################################################

"""GoogleNet xxs from the Choosing Smartly paper: https://arxiv.org/pdf/1707.05733.pdf"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from detectron.core.config import cfg

def add_googlenet_xxs_conv5_body(model):
    
    model.Conv('data', 'conv1-7x7_s2', 3, 64, 7, pad=3, stride=2)
    model.Relu('conv1-7x7_s2', 'conv1-7x7_s2')
    model.MaxPool('conv1-7x7_s2', 'pool1-3x3_s2', kernel=3, pad=0, stride=2)     
    model.LRN('pool1-3x3_s2', 'pool1-norm1', size=5, alpha=0.0001, beta=0.75)
    
    #stop gradient?
    #model.StopGradient('pool1-norm1', 'pool1-norm1')
    model.Conv('pool1-norm1', 'conv2-3x3_reduce', 64, 64, 1, pad=0, stride=1) #input size=?
    model.Relu('conv2-3x3_reduce', 'conv2-3x3_reduce')
    model.Conv('conv2-3x3_reduce', 'conv2-3x3', 64, 192, 3, pad=1, stride=1)
    model.Relu('conv2-3x3', 'conv2-3x3')
    model.LRN('conv2-3x3', 'conv2-norm2', size=5, alpha=0.0001, beta=0.75)
    model.MaxPool('conv2-norm2', 'pool2-3x3_s2', kernel=3, stride=2, pad=0)
    
    #inception 3a
    model.Conv('pool2-3x3_s2', 'inception_3a-1x1', 192, 64, 1, pad=0, stride=1)
    model.Relu('inception_3a-1x1', 'inception_3a-1x1')
    model.Conv('pool2-3x3_s2', 'inception_3a-3x3_reduce', 192, 96, 1, pad=0, stride=1)
    model.Relu('inception_3a-3x3_reduce', 'inception_3a-3x3_reduce')
    model.Conv('inception_3a-3x3_reduce', 'inception_3a-3x3', 96, 128, 3, pad=1, stride=1)
    model.Relu('inception_3a-3x3', 'inception_3a-3x3')
    model.Conv('pool2-3x3_s2', 'inception_3a-5x5_reduce', 192, 16, 1, pad=0, stride=1)
    model.Relu('inception_3a-5x5_reduce', 'inception_3a-5x5_reduce')
    model.Conv('inception_3a-5x5_reduce', 'inception_3a-5x5', 16, 32, 5, pad=2, stride=1)
    model.Relu('inception_3a-5x5', 'inception_3a-5x5')
    model.MaxPool('pool2-3x3_s2', 'inception_3a-pool', kernel=3, stride=1, pad=1)
    model.Conv('inception_3a-pool', 'inception_3a-pool_proj', 192, 32, 1, stride=1, pad=0)
    model.Relu('inception_3a-pool_proj', 'inception_3a-pool_proj')
    model.Concat(['inception_3a-1x1','inception_3a-3x3', 'inception_3a-5x5', 'inception_3a-pool_proj'], 'inception_3a-output')
    
    #inception 3b
    model.Conv('inception_3a-output', 'inception_3b-1x1', 256, 128, 1, pad=0, stride=1)
    model.Relu('inception_3b-1x1', 'inception_3b-1x1')
    model.Conv('inception_3a-output', 'inception_3b-3x3_reduce', 256, 128, 1, pad=0, stride=1)
    model.Relu('inception_3b-3x3_reduce', 'inception_3b-3x3_reduce')
    model.Conv('inception_3b-3x3_reduce', 'inception_3b-3x3', 128, 192, 3, pad=1, stride=1)
    model.Relu('inception_3b-3x3', 'inception_3b-3x3')
    model.Conv('inception_3a-output', 'inception_3b-5x5_reduce', 256, 32, 1, pad=0, stride=1)
    model.Relu('inception_3b-5x5_reduce', 'inception_3b-5x5_reduce')
    model.Conv('inception_3b-5x5_reduce', 'inception_3b-5x5', 32, 96, 5, pad=2, stride=1)
    model.Relu('inception_3b-5x5', 'inception_3b-5x5')
    model.MaxPool('inception_3a-output', 'inception_3b-pool', kernel=3, stride=1, pad=1)
    model.Conv('inception_3b-pool', 'inception_3b-pool_proj', 256, 64, 1, pad=0, stride=1)
    model.Relu('inception_3b-pool_proj', 'inception_3b-pool_proj')
    model.Concat(['inception_3b-1x1', 'inception_3b-3x3', 'inception_3b-5x5', 'inception_3b-pool_proj'], 'inception_3b-output')
    
    model.MaxPool('inception_3b-output', 'pool3-3x3_s2', kernel=3, stride=2, pad=0)
    
    #inception 4a
    model.Conv('pool3-3x3_s2', 'inception_4a-1x1', 480, 192, 1, pad=0, stride=1)
    model.Relu('inception_4a-1x1', 'inception_4a-1x1')
    model.Conv('pool3-3x3_s2', 'inception_4a-3x3_reduce', 480, 96, 1, pad=0, stride=1)
    model.Relu('inception_4a-3x3_reduce', 'inception_4a-3x3_reduce')
    model.Conv('inception_4a-3x3_reduce', 'inception_4a-3x3', 96, 208, 3, pad=1, stride=1)
    model.Relu('inception_4a-3x3', 'inception_4a-3x3')
    model.Conv('pool3-3x3_s2', 'inception_4a-5x5_reduce', 480, 16, 1, pad=0, stride=1)
    model.Relu('inception_4a-5x5_reduce', 'inception_4a-5x5_reduce')
    model.Conv('inception_4a-5x5_reduce', 'inception_4a-5x5', 16, 48, 5, pad=2, stride=1)
    model.Relu('inception_4a-5x5', 'inception_4a-5x5')
    model.MaxPool('pool3-3x3_s2', 'inception_4a-pool', kernel=3, stride=1, pad=1)
    model.Conv('inception_4a-pool', 'inception_4a-pool_proj', 480, 64, 1, pool=0, stride=1)
    model.Relu('inception_4a-pool_proj', 'inception_4a-pool_proj')
    blob_out = model.Concat(['inception_4a-1x1', 'inception_4a-3x3', 'inception_4a-5x5', 'inception_4a-pool_proj'], 'inception_4a-output')
    
    return blob_out, 512, 1. / 16.


def add_googlenet_xxs_roi_fc_head(model, blob_in, dim_in, spatial_scale):
    model.RoIFeatureTransform(
        blob_in,
        'pool5',
        blob_rois='rois',
        method=cfg.FAST_RCNN.ROI_XFORM_METHOD,
        resolution=6,
        sampling_ratio=cfg.FAST_RCNN.ROI_XFORM_SAMPLING_RATIO,
        spatial_scale=spatial_scale
    )
    model.FC('pool5', 'fc6', dim_in * 6 * 6, 4096)
    model.Relu('fc6', 'fc6')
    model.FC('fc6', 'fc7', 4096, 1024)
    blob_out = model.Relu('fc7', 'fc7')
    return blob_out, 1024


