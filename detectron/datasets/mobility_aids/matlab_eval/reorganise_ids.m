function[all_labels, all_detections] = reorganise_ids(labels,detections)

labels_counter = 0;
detection_counter = 0;
image_counter = 0;

all_labels = struct([]);
all_detections = struct([]);

for i=1:length(detections)
    detections_i = detections{i};
    labels_i = labels{i};
    
    for j=1:length(detections_i)
        detections_i(j).detid = detections_i(j).detid + detection_counter;
        detections_i(j).image_id = detections_i(j).image_id + image_counter;
    end
    
    for k=1:length(labels_i.annotations)
        labels_i.annotations(k).id = labels_i.annotations(k).id + labels_counter;
        labels_i.annotations(k).image_id = labels_i.annotations(k).image_id + image_counter;
    end
    
    labels_counter = labels_counter + length(labels_i.annotations);
    detection_counter = detection_counter + length(detections_i);
    image_counter = image_counter + length(labels_i.images);
    
    %append labels
    if isempty(all_labels)
        all_labels = labels_i;
    else
        all_labels.images(end+1:end+length(labels_i.images)) = labels_i.images;
        all_labels.annotations(end+1:end+length(labels_i.annotations)) = labels_i.annotations;
    end
    
    %append detections
    if isempty(all_detections)
        all_detections = detections_i;
    else
        all_detections(end+1:end+length(detections_i)) = detections_i;
    end
    
end

end