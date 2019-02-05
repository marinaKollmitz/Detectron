function all_detections = read_detections(detection_files)

all_detections = {};

for i=1:length(detection_files)

    detection_file = detection_files(i);
    
    fid = fopen(detection_file);
    raw = fread(fid,inf);
    str = char(raw');
    fclose(fid);
    detections = jsondecode(str);
    %add detection id field
    ids = num2cell(1:length(detections));
    [detections(:).detid] = deal(ids{:});
    
    all_detections = [all_detections, detections]; 

end

end