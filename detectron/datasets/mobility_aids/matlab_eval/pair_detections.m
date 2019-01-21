function [assignments] = pair_detections(detections, labels_in, iou_thres)

labels = labels_in;

%add assigned label field
assigned = num2cell(false(1,length(labels)));
[labels(:).assigned] = deal(assigned{:});

% sort detections by decreasing confidence
detection_fields = fieldnames(detections);
det_cell = struct2cell(detections);
sz = size(det_cell);
det_cell = reshape(det_cell, sz(1), [])'; 
if isfield(detections, 'depth')
    det_cell = sortrows(det_cell,-5);
else
    det_cell = sortrows(det_cell,-4);
end
det_cell = reshape(det_cell', sz);
detections = cell2struct(det_cell, detection_fields, 1);

% assign detections to ground truth objects
nd=length(detections); 
assignments = -1*ones(nd,2);

%pair detection with highest confidence first, pair them to the label 
%with the highest iou, if > threshold
for d=1:nd % loop through all detections

    detection = detections(d);
    assignments(d,1) = detection.detid;
    
    % assign detection to ground truth object if any
    bb=detection.bbox;
    ovmax=-inf; %store largest iou for current detection
    jmax = -1;
    
    %find gt for image id
    labels_i = labels([labels.image_id]==detection.image_id);
    
    for j=1:length(labels_i)%loop through all gt labels
        gt = labels_i(j);
        bbgt=gt.bbox;
        
        % calculate intersection over union: 
        % calculations taken from coco eval script
        da=bb(3)*bb(4);
        ga=bbgt(3)*bbgt(4);
        
        w = min(bb(3)+bb(1),bbgt(3)+bbgt(1))-max(bb(1),bbgt(1));
        h = min(bb(4)+bb(2),bbgt(4)+bbgt(2))-max(bb(2),bbgt(2));
        
        if w>0 && h>0
            % compute overlap as area of intersection / area of union
            i = w * h;
            u = da+ga-i;
            ov = i/u;
            if ov>ovmax
                ovmax=ov;
                jmax=j;
            end
        end
    end

    % assign detection to label
    if ovmax>=iou_thres %if there is a label with >threshold iou
        gt = labels_i(jmax);
        if ~gt.assigned %if label has not been assigned
            labels([labels.id]==gt.id).assigned = true; %mark label as assigned
            assignments(d,2) = gt.id; 
        end
    end
end

end