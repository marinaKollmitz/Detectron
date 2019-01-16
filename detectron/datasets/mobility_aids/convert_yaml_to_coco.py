from __future__ import print_function

import yaml
import json
import os

def yaml_to_json(imageset_file, annotation_dir, output_filename, classes):
    
    print("converting annotations for ", imageset_file)
    
    with open(imageset_file, 'r') as stream:
        annotation_files = stream.readlines()
    
    images = []
    annotations = []
    
    #process categories
    class_num = 1
    classes_dict = dict() #map from name to id
    categories = []
    for cl in classes:
        classes_dict[cl] = class_num
        category_dict = dict()
        category_dict["id"] = class_num
        category_dict["name"] = cl
        category_dict["supercategory"] = "none"
        categories.append(category_dict)
        class_num += 1
    
    #process images and annotations
    image_id = 0
    annotation_id = 1
    annotations = []
    
    for annotation_file in annotation_files:
        ann_file = annotation_dir + annotation_file[0:-1]
        ann_file = ann_file + '.yml'
    
        with open(ann_file, 'r') as stream:
            image_dict = dict()
            annotation_dict = dict()
    
            filedata = stream.read()
            annotation_yaml = yaml.load(filedata)

            image_dict['id'] = image_id
            image_dict['width'] = int(annotation_yaml['annotation']['size']['width'])
            image_dict['height'] = int(annotation_yaml['annotation']['size']['height'])
            image_dict['file_name'] = annotation_yaml['annotation']['filename'][0:-3] + 'png'
            images.append(image_dict)
    
            #check if there are labeled instances in the file
            if 'object' in annotation_yaml['annotation']:
                for inst in annotation_yaml['annotation']['object']:
                    annotation_dict = dict()
                    annotation_dict['image_id'] = image_id
                    annotation_dict['id'] = annotation_id
                    annotation_dict['category_id'] = classes_dict[inst['name']]
    
                    bbox = inst['bndbox']
                    xmin = int(bbox['xmin'])
                    ymin = int(bbox['ymin'])
                    xmax = int(bbox['xmax'])
                    ymax = int(bbox['ymax'])
                    annotation_dict['segmentation'] = [[xmin, ymin, xmin, ymax, xmax, ymax, xmax, ymin]]
                    annotation_dict['area'] = (xmax - xmin) * (ymax - ymin)
                    annotation_dict['bbox'] = [xmin, ymin, xmax-xmin, ymax-ymin]
                    annotation_dict['depth'] = bbox['depth']
                    
                    annotation_dict['iscrowd'] = 0
                    
                    annotations.append(annotation_dict)
                    annotation_id += 1
                    
            image_id += 1
            
    print("Num categories: %s" % len(categories))
    print("Num images: %s" % len(images))
    print("Num annotations: %s" % len(annotations))
    
    #create output directory if it does not exist
    directory = os.path.dirname(output_filename)
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    #write json annotation file 
    json_dict = dict()
    json_dict['images'] = images
    json_dict['type'] = "instances"
    json_dict['annotations'] = annotations
    json_dict['categories'] = categories
    
    with open(output_filename, 'w') as outfile:
        json.dump(json_dict, outfile) 
        print("wrote annotations to %s" % (output_filename))
