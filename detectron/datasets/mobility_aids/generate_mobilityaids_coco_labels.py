import os 
mobilityaids_dir = os.path.abspath(__file__ + "/../../data/mobility-aids/")
from convert_yaml_to_coco import yaml_to_json

import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Generate coco labels from VOC2007 yml annotations')
    parser.add_argument(
        '--with_InOutDoor',
        dest='with_inoutdoor',
        help='generate RGB training annotations with additional InOutDoor examples',
        action='store_true'
    )
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()

    imageset_files = [os.path.join(mobilityaids_dir, "ImageSets/TrainSet_DepthJet.txt"),
                      os.path.join(mobilityaids_dir, "ImageSets/TestSet1.txt"),
                      os.path.join(mobilityaids_dir, "ImageSets/TestSet2_seq1.txt"),
                      os.path.join(mobilityaids_dir, "ImageSets/TestSet2_seq2.txt"),
                      os.path.join(mobilityaids_dir, "ImageSets/TestSet2_seq3.txt"),
                      os.path.join(mobilityaids_dir, "ImageSets/TestSet2_seq4.txt"),
                      os.path.join(mobilityaids_dir, "ImageSets/TrainSet_RGB.txt"),
                      os.path.join(mobilityaids_dir, "ImageSets/TestSet1.txt"),
                      os.path.join(mobilityaids_dir, "ImageSets/TestSet2_seq1.txt"),
                      os.path.join(mobilityaids_dir, "ImageSets/TestSet2_seq2.txt"),
                      os.path.join(mobilityaids_dir, "ImageSets/TestSet2_seq3.txt"),
                      os.path.join(mobilityaids_dir, "ImageSets/TestSet2_seq4.txt")]
    
    annotation_dirs = [os.path.join(mobilityaids_dir, "Annotations_DepthJet/"),
                       os.path.join(mobilityaids_dir, "Annotations_DepthJet/"),
                       os.path.join(mobilityaids_dir, "Annotations_DepthJet_TestSet2/"),
                       os.path.join(mobilityaids_dir, "Annotations_DepthJet_TestSet2/"),
                       os.path.join(mobilityaids_dir, "Annotations_DepthJet_TestSet2/"),
                       os.path.join(mobilityaids_dir, "Annotations_DepthJet_TestSet2/"),
                       os.path.join(mobilityaids_dir, "Annotations_RGB/"),
                       os.path.join(mobilityaids_dir, "Annotations_RGB/"),
                       os.path.join(mobilityaids_dir, "Annotations_RGB_TestSet2/"),
                       os.path.join(mobilityaids_dir, "Annotations_RGB_TestSet2/"),
                       os.path.join(mobilityaids_dir, "Annotations_RGB_TestSet2/"),
                       os.path.join(mobilityaids_dir, "Annotations_RGB_TestSet2/")]
    
    output_filenames = [os.path.join(mobilityaids_dir, "annotations/train_DepthJet.json"),
                        os.path.join(mobilityaids_dir, "annotations/test_DepthJet.json"),
                        os.path.join(mobilityaids_dir, "annotations/test2_DepthJet_seq1.json"),
                        os.path.join(mobilityaids_dir, "annotations/test2_DepthJet_seq2.json"),
                        os.path.join(mobilityaids_dir, "annotations/test2_DepthJet_seq3.json"),
                        os.path.join(mobilityaids_dir, "annotations/test2_DepthJet_seq4.json"),
                        os.path.join(mobilityaids_dir, "annotations/train_RGB.json"),
                        os.path.join(mobilityaids_dir, "annotations/test_RGB.json"),
                        os.path.join(mobilityaids_dir, "annotations/test2_RGB_seq1.json"),
                        os.path.join(mobilityaids_dir, "annotations/test2_RGB_seq2.json"),
                        os.path.join(mobilityaids_dir, "annotations/test2_RGB_seq3.json"),
                        os.path.join(mobilityaids_dir, "annotations/test2_RGB_seq4.json")]
    
    classes = ["person", "crutches", "walking_frame", "wheelchair", "push_wheelchair"]
    
    for i in range(len(imageset_files)):
        imageset_file = imageset_files[i]
        annotation_dir = annotation_dirs[i]
        output_filename = output_filenames[i]
        
        if not os.path.exists(output_filename):
            #get annotation files from imageset file
            with open(imageset_file) as stream:
                annotation_files = stream.readlines()
            
            annotation_files = [annotation_dir + filename.strip() + '.yml' for filename in annotation_files]
            
            yaml_to_json(annotation_files, output_filename, classes)
        else:
            print "%s exists, no need to generate" % output_filename
    
    # if desired, generate RGB training set with mobilityaids and additional
    # InOutDoorPeople examples. Requires to download the InOutDoorPeople dataset
    # and linking the images, like described in the mobilityaids howto:
    # https://github.com/marinaKollmitz/DetectronDistance/blob/master/MOBILITYAIDS_HOWTO.md
    if args.with_inoutdoor:
        
        #DepthJet annotations
        imageset_files_DJ = [os.path.join(mobilityaids_dir, "ImageSets/TrainSet_DepthJet.txt"),
                             os.path.join(mobilityaids_dir, "ImageSets/TrainAdditional_InOutDoor.txt")]
        
        annotation_dirs_DJ = [os.path.join(mobilityaids_dir, "Annotations_DepthJet/"),
                             os.path.join(mobilityaids_dir, "Annotations_InOutDoor_DepthJet/")]
        
        output_filename_DJ = os.path.join(mobilityaids_dir, "annotations/train_DepthJet_w_InOutDoor.json")
        
        annotation_files_DJ = []
        
        if not os.path.exists(output_filename_DJ):
            
            for i in range(len(imageset_files_DJ)):
            
                #get annotation files from imageset file
                with open(imageset_files_DJ[i]) as stream:
                    annotation_files = stream.readlines()
            
                annotation_files = [annotation_dirs_DJ[i] + filename.strip() + '.yml' for filename in annotation_files]
                annotation_files_DJ.extend(annotation_files)
                
            yaml_to_json(annotation_files_DJ, output_filename_DJ, classes)
        
        else:
            print "%s exists, no need to generate" % output_filename
            
        imageset_files_RGB = [os.path.join(mobilityaids_dir, "ImageSets/TrainSet_RGB.txt"),
                              os.path.join(mobilityaids_dir, "ImageSets/TrainAdditional_InOutDoor.txt")]
        
        #RGB annotations
        annotation_dirs_RGB = [os.path.join(mobilityaids_dir, "Annotations_RGB/"),
                               os.path.join(mobilityaids_dir, "Annotations_InOutDoor_RGB/")]
        
        output_filename_RGB = os.path.join(mobilityaids_dir, "annotations/train_RGB_w_InOutDoor.json")
        
        annotation_files_RGB = []
        
        if not os.path.exists(output_filename_RGB):
            
            for i in range(len(imageset_files_RGB)):
            
                #get annotation files from imageset file
                with open(imageset_files_RGB[i]) as stream:
                    annotation_files = stream.readlines()
            
                annotation_files = [annotation_dirs_RGB[i] + filename.strip() + '.yml' for filename in annotation_files]
                annotation_files_RGB.extend(annotation_files)
                
            yaml_to_json(annotation_files_RGB, output_filename_RGB, classes)
        
        else:
            print "%s exists, no need to generate" % output_filename