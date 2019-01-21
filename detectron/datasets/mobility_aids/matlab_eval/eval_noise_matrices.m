function [observation_model, meas_cov] = eval_noise_matrices(assignments, detections, labels)

num_labels = length(labels.annotations);
num_dets=length(assignments); 
num_classes = length(labels.categories);

classes = cell(1,length(labels.categories));
for i=1:length(labels.categories)
    classes{labels.categories(i).id} = labels.categories(i).name;
end

% to speed up the evaluation, make a structure array for the detections 
% and labels
for i=1:num_dets
    detection = detections(i);
    detection_array(detection.detid) = detection;
end
for i=1:num_labels
    label = labels.annotations(i);
    label.assigned = false; %add assigned field
    labels_array(label.id) = label;
end

num_dets_class = zeros(num_classes+1,1);

%confusion matrix: rows(ind1): labels    cols(ind2): detections
confusion_matrix = zeros(num_classes+1, num_classes+1);

bbox_per_image = 500; %total number of bboxes classified for each image
total_classified_bboxes = bbox_per_image * length(labels.images);

depth_diffs = nan(num_dets,1);
x_diffs = nan(num_dets,1);
y_diffs = nan(num_dets,1);

for d=1:num_dets % loop through all detections
    
    assignment = assignments(d,:);
    detection = detection_array(assignment(1));
    
    num_dets_class(detection.category_id + 1) = num_dets_class(detection.category_id + 1) + 1;
    
    if assignment(2) >= 0
        
        label = labels_array(assignment(2));
        labels_array(assignment(2)).assigned = true;
        
        %count detection in confusion matrix
        label_ind = label.category_id + 1;
        det_ind = detection.category_id + 1;
        confusion_matrix(label_ind,det_ind) = confusion_matrix(label_ind,det_ind) + 1;
        
        if isfield(detection, 'depth')
            if(label.depth > 0)
                depth_gt = label.depth;
                depth_det = detection.depth;
                
                bb_label = label.bbox;
                bb_det = detection.bbox;
                
                %%%gt coordinates
                bbox_xmin = bb_label(1);
                bbox_ymin = bb_label(2);
                bbox_width = bb_label(3);
                bbox_height = bb_label(4);
                
                im_x_gt = bbox_xmin + bbox_width/2;
                im_y_gt = bbox_ymin + bbox_height/2;
                
                %%%detection coordinates
                bbox_xmin = bb_det(1);
                bbox_ymin = bb_det(2);
                bbox_width = bb_det(3);
                bbox_height = bb_det(4);
                
                im_x_det = bbox_xmin + bbox_width/2;
                im_y_det = bbox_ymin + bbox_height/2;
                
                depth_diffs(d) = depth_gt - depth_det;
                x_diffs(d) = im_x_gt - im_x_det;
                y_diffs(d) = im_y_gt - im_y_det;
                
            end
        end
    else %there is no label for this detection (false positive)
        label_ind = 1; %label is background
        det_ind = detection.category_id + 1;
        confusion_matrix(label_ind,det_ind) = confusion_matrix(label_ind,det_ind) + 1;
    end
end

%check if labels were not assigned (false negative)
for j=1:num_labels%loop through all gt labels
    label = labels_array(j);
    if label.assigned ~= true
        label_ind = label.category_id + 1;
        det_ind = 1; %detection is background
        confusion_matrix(label_ind,det_ind) = confusion_matrix(label_ind,det_ind) + 1;
        num_dets_class(det_ind) = num_dets_class(det_ind) + 1;
    end
end

%calculate covariance from matched detections
depth_diffs = depth_diffs(~isnan(depth_diffs)); 
x_diffs = x_diffs(~isnan(x_diffs)); 
y_diffs = y_diffs(~isnan(y_diffs)); 

meas_cov = cov([x_diffs,y_diffs,depth_diffs]);

%calculate number of true negatives from total number of bounding boxes
num_false_positives = sum(confusion_matrix(1,:));
confusion_matrix(1,1) = total_classified_bboxes - num_false_positives;

%add dirichlet prior
confusion_matrix = confusion_matrix + ones(num_classes+1, num_classes+1);

%normalize confusion matrix by column to get observation model
observation_model = zeros(num_classes+1, num_classes+1);
for i = 1:num_classes+1
    observation_model(i,:) = confusion_matrix(i,:) / sum(confusion_matrix(i,:));
end

end