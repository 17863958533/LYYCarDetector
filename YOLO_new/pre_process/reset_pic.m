% 将倒置的图片转换过来

clear
clc

load('carDatasetGroundTruth.mat','carDataset');
carDataset.imageFilename = fullfile(pwd,carDataset.imageFilename);
pic_name = carDataset.imageFilename;
count = 0;
for i = 1:length(pic_name)
    im = imread(pic_name(i,1));
    size_pic = size(im);
    if size_pic(1) > size_pic(2)
        text1=pic_name(i,1)
%         fprintf(text1(end-21:end));
        count = count+1;
    end
    
    
end
count