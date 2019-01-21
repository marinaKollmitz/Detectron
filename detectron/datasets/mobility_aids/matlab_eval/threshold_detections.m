function thresh_detections = threshold_detections(detections, thresholds)

nd=length(detections);
valid_det_i = 1;

for d=1:nd % loop through all detections
    
    detection = detections(d);
    
    if detection.score > thresholds(detection.category_id)
        thresh_detections(valid_det_i) = detection;
            
        valid_det_i = valid_det_i + 1;
    end

end