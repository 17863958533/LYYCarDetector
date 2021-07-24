% 抽取用于训练的数据集（原数据集过大）
clear
clc

load('carDatasetGroundTruth_raw.mat') % 载入原数据集


data = carDataset.vehicle;

for i = 1:9879
    width = data{i,1}(:,3) - data{i,1}(:,1);
    height = data{i,1}(:,4) - data{i,1}(:,2);
    data{i,1}(:,3:4) = [width, height];
    data{i,1}(:,1:2) = [data{i,1}(:,1)+1, data{i,1}(:,2)+1];
end

carDataset.vehicle = data;

save('carDatasetGroundTruth.mat', 'carDataset')

% 取前1000张图片
imageFilename = carDataset.imageFilename(1:1000,:);
vehicle = carDataset.vehicle(1:1000, :);
carDataset = table(imageFilename, vehicle);
save('carDatasetGroundTruth_1000.mat', 'carDataset')