function [rec,prec,ap,rec_at_ap,thres_at_ap] = eval_image_aps(assignments, detections, labels, tracking)

num_labels = length(labels);
num_dets=length(assignments); 

%detection has a label
tp = (assignments(:,2)>=0); 

%detection does not have a label
fp = (assignments(:,2)<0); 

% compute precision/recall
if tracking
    %only calculate precision and recall
    fp = sum(fp);
    tp = sum(tp);
    
    rec_at_ap=tp/num_labels;
    ap=tp./(fp+tp);
    
    %don't evaluate for tracking
    prec = 0;
    rec = 0;
    thres_at_ap = -1;

else    
    fp=cumsum(fp);
    tp=cumsum(tp);
    
    rec=tp/num_labels;
    prec=tp./(fp+tp);
    
    ap=VOCap(rec,prec);

    if num_dets > 0
        residuals = abs(prec - ap);
        [~, min_ind] = min(residuals);
        rec_at_ap = rec(min_ind);
        det = detections([detections.detid]==assignments(min_ind,1));
        thres_at_ap = det.score;
    else
        rec_at_ap = -1;
        thres_at_ap = -1;
    end
end

end