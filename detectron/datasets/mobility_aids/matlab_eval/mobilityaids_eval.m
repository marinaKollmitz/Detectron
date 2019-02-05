function mobilityaids_eval(json_labels_files, res_files, output_dir, tracking)

%if tracking is not specified, set it to false
if nargin<4
   tracking = false;
end

write_results = true;

labels = get_json_labels(json_labels_files);
detections = read_detections(res_files);

[labels, detections] = reorganise_ids(labels, detections);

classes = cell(1,length(labels.categories));
for i=1:length(labels.categories)
    classes{labels.categories(i).id} = labels.categories(i).name;
end

dist_thresholds = (0:0.025:3.5);
thresholds = zeros(length(classes),1);
aps = zeros(length(classes),1);
recs = zeros(length(classes),1);
depth_aps = zeros(length(dist_thresholds), length(classes));
depth_recs = zeros(length(dist_thresholds), length(classes));

all_gt_depths = [];
all_depth_errors = [];

for cl_i=1:length(classes)
    cla = classes{cl_i};
    fprintf("evaluating %s\n", cla);

    %get all labels for current category id
    labels_cl = labels.annotations([labels.annotations.category_id]==cl_i);

    %get all detections for current category id
    detections_cl = detections([detections.category_id]==cl_i);

    %pair detections to labels
    assignments = pair_detections(detections_cl, labels_cl, 0.5);

    %evaluate image AP score
    [rec,prec,ap,rec_at_ap,thres_at_ap] = eval_image_aps(assignments, detections_cl, labels_cl, tracking);

    aps(cl_i) = ap;
    recs(cl_i) = rec_at_ap;
    thresholds(cl_i) = thres_at_ap;

    %evaluate depth errors
    [gt_depths, depth_errors] = eval_depth_errors(assignments, detections_cl, labels_cl);
    all_gt_depths = [all_gt_depths; gt_depths];
    all_depth_errors = [all_depth_errors; depth_errors];

    %evaluate depth error APs
    [distance_aps, distance_recs] = depth_error_aps(assignments, gt_depths, depth_errors, labels_cl, dist_thresholds, tracking);
    depth_aps(:,cl_i) = distance_aps;
    depth_recs(:,cl_i) = distance_recs;

end

% Print Results
if tracking
    fprintf("\nmean depth error: %.5f m\n", mean(all_depth_errors(all_depth_errors>0)));

    %Print precision and recall performance
    fprintf("\ndetection results\n\n");
    %output precision recall evaluation
    fprintf("%18s %12s %12s %12s %12s\n" , "class", "im_prec", "im_rec", "p@0.5m", "rec@0.5m");

    for cl_i=1:length(classes)
        cla = classes{cl_i};
        fprintf("%18s %12.5f %12.5f %12.5f %12.5f\n", cla, aps(cl_i), recs(cl_i), depth_aps(dist_thresholds==0.5,cl_i), depth_recs(dist_thresholds==0.5,cl_i));
    end

    fprintf("%18s\n", "----");
    depth_maps = mean(depth_aps,2);
    depth_mrecs = mean(depth_recs,2);
    fprintf("%18s %12.5f %12.5f %12.5f %12.5f\n", "mAP", mean(aps), mean(recs), depth_maps(dist_thresholds==0.5), depth_mrecs(dist_thresholds==0.5)); 

    if write_results
        %output precision recall evaluation
        file = fopen(fullfile(output_dir, 'precision-recall.txt'), 'w');
        fprintf(file, "%18s %12s %12s %12s %12s\n" , "class", "im_prec", "im_rec", "p@0.5m", "rec@0.5m");

        for cl_i=1:length(classes)
            cla = classes{cl_i};
            fprintf(file, "%18s %12.5f %12.5f %12.5f %12.5f\n", cla, aps(cl_i), recs(cl_i), depth_aps(dist_thresholds==0.5,cl_i), depth_recs(dist_thresholds==0.5,cl_i));
        end
        
        fprintf(file, "%18s\n", "----");
        
        depth_maps = mean(depth_aps,2);
        depth_mrecs = mean(depth_recs,2);
        fprintf(file, "%18s %12.5f %12.5f %12.5f %12.5f\n", "mAP", mean(aps), mean(recs), depth_maps(dist_thresholds==0.5), depth_mrecs(dist_thresholds==0.5)); 
        
        fprintf(file, "\n mean depth error: %.5f m\n", mean(all_depth_errors(all_depth_errors>0)));
            
        fclose(file);
    end
    
else
    %threshold detections
    thresh_dets = threshold_detections(detections, thresholds);

    %pair detections to labels, without considering the class labels
    assignments = pair_detections(thresh_dets, labels.annotations, 0.5);

    %calculate observation model and measurement covariance
    [observation_model, meas_cov] = eval_noise_matrices(assignments, thresh_dets, labels);

    fprintf("\ndetection results\n\n");

    %output precision recall evaluation
    fprintf("%18s %12s %12s %12s \n" , "class", "im_ap", "ap@0.25m", "ap@0.5m");

    for cl_i=1:length(classes)
        cla = classes{cl_i};
        fprintf("%18s %12.5f %12.5f %12.5f\n", cla, aps(cl_i), depth_aps(dist_thresholds==0.25,cl_i), depth_aps(dist_thresholds==0.5,cl_i));
    end

    fprintf("%18s\n", "----");
    depth_maps = mean(depth_aps,2);
    fprintf("%18s %12.5f %12.5f %12.5f\n", "mAP", mean(aps), depth_maps(dist_thresholds==0.25), depth_maps(dist_thresholds==0.5)); 

    % calculate depth regression error
    [gt_intervals, depth_error_quantiles, outliers] = depth_reg_quantiles(all_gt_depths, all_depth_errors);

    if write_results

        fprintf("\nwriting results to folder: \n %s \n\n", output_dir);

        %output precision recall evaluation
        file = fopen(fullfile(output_dir, 'AP.txt'), 'w');
        fprintf(file, "%18s %12s %12s %12s \n" , "class", "im_ap", "ap@0.25m", "ap@0.5m");

        for cl_i=1:length(classes)
            cla = classes{cl_i};
            fprintf(file, "%18s %12.5f %12.5f %12.5f\n", cla, aps(cl_i), depth_aps(dist_thresholds==0.25,cl_i), depth_aps(dist_thresholds==0.5,cl_i));
        end

        depth_maps = mean(depth_aps,2);
        fprintf(file, "%18s %12.5f %12.5f %12.5f\n", "mAP", mean(aps), depth_maps(dist_thresholds==0.25), depth_maps(dist_thresholds==0.5));

        fclose(file);

        % write thresholds at AP
        T = table(classes',thresholds);
        writetable(T,fullfile(output_dir, 'AP_thresholds.txt'), 'WriteVariableNames', false)

        %observation model
        dlmwrite(fullfile(output_dir, 'observation_model.txt'), observation_model)

        %measurement covariance
        dlmwrite(fullfile(output_dir, 'meas_cov.txt'), meas_cov)

        %depth regression error
        file = fopen(fullfile(output_dir, 'depth_reg_error.txt'), 'w');
        fprintf(file, "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n", "int_min", "int_max", "quant_0.5", "quant_0.75", "quant_0.25", "quant_0.95", "quant_0.05", "max_outlier1", "max_outlier2", "max_outlier3");
        for i=1:length(gt_intervals)
            interval = gt_intervals(i,:);
            quantiles = depth_error_quantiles(i,:);
            out = outliers(i,:);

            fprintf(file, "%.1f\t%.1f\t%.5f\t%.5f\t%.5f\t%.5f\t%.5f\t%.5f\t%.5f\t%.5f\n", interval(1), interval(2), quantiles, out);
        end

        fclose(file);

        %save distance map to txt file
        distance_map = sum(depth_aps,2)/length(classes);
        A = [transpose(dist_thresholds), distance_map]';

        fileID = fopen(fullfile(output_dir, 'dist_maps.txt'),'w');
        fprintf(fileID,'%s\t%s\n','d','map(d)');
        fprintf(fileID,'%.4f\t%.4f\n',A);
        fclose(fileID);

        fprintf("done\n\n");

    end

end

end
