import argparse
import sys
from detectron.datasets.mobility_aids.convert_yaml_to_coco import yaml_to_json

def parse_args():
    parser = argparse.ArgumentParser(description='Convert dataset')
    parser.add_argument(
        '--imageset', help="imageset text file", default=None, type=str)
    parser.add_argument(
        '--outfile', help="output (.json) file name", default=None, type=str)
    parser.add_argument(
        '--annodir', help="data dir for annotations to be converted",
        default=None, type=str)
    parser.add_argument(
        '--classes', help="class names in dataset", default=None, type=str, 
        nargs='+')
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    yaml_to_json(args.imageset, args.annodir, args.outfile, args.classes)