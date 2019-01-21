function [distance_maps, distance_recs] = depth_error_aps(assignments, gt_depths, depth_errors, labels, dist_thresholds, tracking)

num_labels = length(labels);

%true positives: detection has a label
tp = (assignments(:,2)>=0); 

%false positive: detection does not have a label
fp = (assignments(:,2)<0); 

%filter tp detections for which there is no depth label
tp_depths = gt_depths;
tp_depths(~tp) = 0;
ignore_det = tp_depths < 0;

% find number of labels (ignore labels without gt depth which were paired )
num_valid_labels = num_labels - sum(ignore_det);

% remove ignore entries for which we dont have a distance label
tp = tp(ignore_det == 0);
fp = fp(ignore_det == 0);
depth_diffs = depth_errors(ignore_det == 0);

% get all depth distances for valid tp detections
%depth_diffs_tp = depth_diffs(depth_diffs > 0);
%num_diffs = length(depth_diffs_tp);
%sorted_depth_diffs = sort(depth_diffs_tp);

APs = zeros(length(dist_thresholds),1);
RECs = zeros(length(dist_thresholds),1);
for i=1:length(dist_thresholds)
    dist_threshold = dist_thresholds(i);
    
    %select true positives for distance threshold
    thres_tp = tp;
    thres_tp(depth_diffs>dist_threshold) = 0;
    
    if tracking
        %only calculate precision and recall for this threshold
        fp_sum = sum(fp);
        tp_sum = sum(thres_tp);

        rec=tp_sum/num_labels;
        ap=tp_sum/(fp_sum+tp_sum);

        RECs(i) = rec;
    else
        %calculate precision recall and ap for this threshold
        fp_cum=cumsum(fp);
        tp_cum=cumsum(thres_tp);

        rec=tp_cum/num_valid_labels;
        prec=tp_cum./(fp_cum+tp_cum);

        ap=VOCap(rec,prec);
        
    end
    
    APs(i) = ap;
end

distance_maps = APs;
distance_recs = RECs;

end