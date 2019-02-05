function labels = get_json_labels(label_files)

labels = {};

for i=1:length(label_files)
    label_file = label_files(i);
    %parse label file
    fid = fopen(label_file);
    raw = fread(fid,inf);
    str = char(raw');
    fclose(fid);
    labels = [labels, jsondecode(str)];

end

end