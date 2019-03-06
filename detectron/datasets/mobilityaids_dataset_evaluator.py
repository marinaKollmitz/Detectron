##############################################################################
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
#
# Author: Marina Kollmitz
#
##############################################################################

"""mobilityaids dataset evaluation interface."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import logging
import os

from detectron.core.config import cfg
from detectron.datasets.dataset_catalog import get_ann_fn
from detectron.datasets.json_dataset_evaluator import _write_coco_bbox_results_file

logger = logging.getLogger(__name__)


def evaluate_boxes(
    json_dataset,
    all_boxes,
    all_depths,
    output_dir,
    cleanup=False,
    use_matlab=True):
    
    output_dir = os.path.abspath(output_dir)
    
    res_file = os.path.join(
        output_dir, 'bbox_' + json_dataset.name + '_results.json'
    )
    
    _write_coco_bbox_results_file(json_dataset, all_boxes, all_depths, res_file)
    
    if use_matlab:
        _do_matlab_box_eval(json_dataset, res_file, output_dir)
    else:
        logger.warn("no python evaluator available for the mobilityaids dataset.")
    if cleanup:
        os.remove(res_file)
        
    #TODO return evaluated APs for mobilityaids
    return None

def evaluate_tracking(
    res_files,
    json_datasets,
    output_dir,
    use_matlab=True):
    
    if use_matlab:
        _do_matlab_tracking_eval(json_datasets, res_files, output_dir)
    else:
        logger.warn("no python evaluator available for the mobilityaids dataset.")
    

def _do_matlab_box_eval(json_dataset, res_file, output_dir):
    import subprocess
    
    json_file = get_ann_fn(json_dataset.name)
    
    #TODO possible to format this easier?
    json_files_string = ("[string(\'" + ")\',string(\'".join([json_file]) + "\')]")
    res_files_string = ("[string(\'" + ")\',string(\'".join([res_file]) + "\')]")
    
    logger.info('---------------------------------------------------------')
    logger.info('Computing results with the mobilityaids MATLAB eval code.')
    logger.info('---------------------------------------------------------')
    
    path = os.path.join(
        cfg.ROOT_DIR, 'detectron', 'datasets', 'mobility_aids', 'matlab_eval')
    cmd = 'cd {} && '.format(path)
    cmd += '{:s} -nodisplay -nodesktop '.format(cfg.MATLAB)
    cmd += '-r "dbstop if error; '
    cmd += 'mobilityaids_eval({:s},{:s},\'{:s}\'); quit;"' \
       .format(json_files_string, res_files_string, output_dir)
    logger.info('Running:\n{}'.format(cmd))
    subprocess.call(cmd, shell=True)
    
def _do_matlab_tracking_eval(json_datasets, res_files, output_dir):
    import subprocess
    
    json_files = [get_ann_fn(json_dataset.name) for json_dataset in json_datasets]
    
    #TODO possible to format this easier?
    json_files_string = ("[string(\'" + "\'),string(\'".join(json_files) + "\')]")
    res_files_string = ("[string(\'" + "\'),string(\'".join(res_files) + "\')]")
    
    logger.info('------------------------------------------------------------------')
    logger.info('Computing tracking results with the mobilityaids MATLAB eval code.')
    logger.info('------------------------------------------------------------------')
    
    path = os.path.join(
        cfg.ROOT_DIR, 'detectron', 'datasets', 'mobility_aids', 'matlab_eval')
    cmd = 'cd {} && '.format(path)
    cmd += '{:s} -nodisplay -nodesktop '.format(cfg.MATLAB)
    cmd += '-r "dbstop if error; '
    cmd += 'mobilityaids_eval({:s},{:s},\'{:s}\',true); quit;"' \
       .format(json_files_string, res_files_string, output_dir)
    logger.info('Running:\n{}'.format(cmd))
    subprocess.call(cmd, shell=True)