function labels = get_json_labels(label_file)
%parse label file
fid = fopen(label_file);
raw = fread(fid,inf);
str = char(raw');
fclose(fid);
labels = jsondecode(str);

end