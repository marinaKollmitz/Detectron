function [gt_intervals, depth_error_quantiles, max_outliers] = depth_reg_quantiles(gt_depths, depth_errors)

depth_errors = depth_errors(gt_depths > 0); 
gt_depths = gt_depths(gt_depths > 0);

num_bins = 8;
bin_width = 1;

depth_error_quantiles = zeros(num_bins,5);
max_outliers = -1*ones(num_bins,3);

%get gt depth intervals
gt_intervals = transpose([(0:num_bins-1)*bin_width; (1:num_bins)*bin_width]);
gt_intervals(num_bins,2) = inf;

for i = 1:length(gt_intervals)
    
    gt_interval = gt_intervals(i,:);
    
    %find examples with gt error within bin
    in_bins = depth_errors(gt_depths > gt_interval(1) & gt_depths < gt_interval(2));
           
    %get confidence interval
    if ~isempty(in_bins)
        quantiles = quantile(in_bins, [0.5, 0.75, 0.25, 0.95, 0.05]);
        depth_error_quantiles(i,:) = quantiles;
        interval_outliers = in_bins(in_bins>quantiles(end-1));
        interval_outliers = sort(interval_outliers, 'descend');
        interval_outliers = [interval_outliers; -1*ones(3-length(interval_outliers),1)];
        max_out = transpose(interval_outliers(1:3));
        max_outliers(i,1:length(max_out)) = max_out;
    end
    
end

end