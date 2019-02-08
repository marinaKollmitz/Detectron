from detectron.core.config import assert_and_infer_cfg
from detectron.core.config import cfg
from detectron.core.config import merge_cfg_from_file
from detectron.core.config import merge_cfg_from_list
from detectron.utils.logging import setup_logging

from detectron.core.track_engine import run_tracking

from caffe2.python import workspace

import argparse
import sys
import os
import time
import pprint

def parse_args():
    parser = argparse.ArgumentParser(description='Test the multiclass tracking modul on detections')
    
    parser.add_argument(
        '--cfg',
        dest='cfg_file',
        help='optional config file',
        default=None,
        type=str
    )
    parser.add_argument(
        '--timestep',
        dest='timestep',
        help='(optional) time step between frames, default 0.06s',
        default=0.06,
        type=float
    )
    parser.add_argument(
        '--ekf-only',
        dest='ekf_only',
        help='(optional) only evaluate ekf without hmm',
        action='store_true'
    )
    parser.add_argument(
        '--no-filtering',
        dest='no_filtering',
        help='(optional) evaluate detections without filtering',
        action='store_true'
    )
    parser.add_argument(
        '--visualize',
        dest='viz',
        help='(optional) visualize tracking with matplotlib',
        action='store_true'
    )
    parser.add_argument(
        '--step', 
        dest='step', 
        help='(optional) wait for key stroke after each observation', 
        action='store_true'
    )
    parser.add_argument(
        'opts',
        help='See detectron/core/config.py for all options',
        default=None,
        nargs=argparse.REMAINDER
    )
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    
    args = parser.parse_args()

    return args

if __name__ == '__main__':
    workspace.GlobalInit(['caffe2', '--caffe2_log_level=0'])
    logger = setup_logging(__name__)

    args = parse_args()
    
    logger.info('Called with args:')
    logger.info(args)
    
    if args.cfg_file is not None:
        merge_cfg_from_file(args.cfg_file)
    if args.opts is not None:
        merge_cfg_from_list(args.opts)
    
    assert_and_infer_cfg()
    logger.info('Test tracking with config:')
    logger.info(pprint.pformat(cfg))
    
    #visualization options
    step = args.step
    viz = args.viz
    
    #use hmm
    use_hmm = not args.ekf_only
    
    while not os.path.exists(cfg.TEST.WEIGHTS) and args.wait:
        logger.info('Waiting for \'{}\' to exist...'.format(cfg.TEST.WEIGHTS))
        time.sleep(10)

    run_tracking(
        cfg.TRACK.VALIDATION_DATASET,
        cfg.TRACK.DATASETS,
        cfg.TEST.WEIGHTS,
        args.timestep,
        use_hmm=use_hmm,
        no_filtering=args.no_filtering,
        visualize=args.viz,
        step=args.step)