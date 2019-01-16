import os 
mobilityaids_dir = os.path.abspath(__file__ + "/../../data/mobility-aids/")
from convert_yaml_to_coco import yaml_to_json

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
    
    yaml_to_json(imageset_file, annotation_dir, output_filename, classes)