function [gt_depths, depth_errors] = eval_depth_errors(assignments, detections, labels)

num_labels = length(labels);
num_dets=length(assignments); 

%find depth errors
depth_errors = -1*ones(num_dets,1);
gt_depths = -1*ones(num_dets,1);

% to speed up the evaluation, make a structure array for the detections 
% and labels
for i=1:num_dets
    detection = detections(i);
    detection_array(detection.detid) = detection;
end
for i=1:num_labels
    label = labels(i);
    labels_array(label.id) = label;
end

%evaluate depth errors
for d=1:num_dets
    
    assignment = assignments(d,:);
    detection = detection_array(assignment(1));
    
    if assignment(2) >= 0
        
        label = labels_array(assignment(2));
        
        if isfield(detection, 'depth')
            depthgt = label.depth;
            depthdet = detection.depth;
            
            if(label.depth > 0)
                depth_errors(d) = abs(depthdet-depthgt);
                gt_depths(d) = depthgt;
            end
        end
        
    end
    
end