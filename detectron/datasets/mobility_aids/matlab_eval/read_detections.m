function detections = read_detections(detection_file)

fid = fopen(detection_file);
raw = fread(fid,inf);
str = char(raw');
fclose(fid);
detections = jsondecode(str);
%add detection id field
ids = num2cell(1:length(detections));
[detections(:).detid] = deal(ids{:});

end