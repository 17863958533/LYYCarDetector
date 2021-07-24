clear
clc

doTraining = true; % 是否进行训练

% 解压数据
% 这一部分8 9两行的数据是大训练集的数据，现在本程序使用的是小训练集的数据,这两部分二选一
% data = load('./data/carDatasetGroundTruth.mat');
% vehicleDataset = data.carDataset; % table型，包含文件路径和groundTruth
data = load('./VehicleDetection/train/data/vehicleDatasetGroundTruth.mat');
vehicleDataset = data.vehicleDataset; % table型，包含文件路径和groundTruth

% 添加绝对路径至vehicleDataset中
vehicleDataset.imageFilename = fullfile([pwd, '/VehicleDetection/train/data/'],vehicleDataset.imageFilename);

% 显示数据集中的一个图像，以了解它包含的图像的类型。
vehicleDataset(1:4,:) % 显示部分数据情况

% 将数据集分成两部分：一个是用于训练检测器的训练集，一个是用于评估检测器的测试集。
% 选择 70% 的数据进行训练，其余数据用于评估。
rng(0); % 控制随机数生成
shuffledIndices = randperm(height(vehicleDataset));
idx = floor(0.7 * length(shuffledIndices) );
trainingDataTbl = vehicleDataset(shuffledIndices(1:idx),:);
testDataTbl = vehicleDataset(shuffledIndices(idx+1:end),:);

% 保存数据和标签
imdsTrain = imageDatastore(trainingDataTbl{:,'imageFilename'}); % 路径
bldsTrain = boxLabelDatastore(trainingDataTbl(:,'vehicle')); % 真实框和类别

imdsTest = imageDatastore(testDataTbl{:,'imageFilename'});
bldsTest = boxLabelDatastore(testDataTbl(:,'vehicle'));

% 整理训练测试集
trainingData = combine(imdsTrain,bldsTrain); % 联合文件路径和真实框
testData = combine(imdsTest,bldsTest);


% 显示数据
data = read(trainingData); % data包括图片数据、真实框坐标、类别
I = data{1};
bbox = data{2};
annotatedImage = insertShape(I,'Rectangle',bbox); % 在数据矩阵中标出真实框
annotatedImage = imresize(annotatedImage,2);
figure
imshow(annotatedImage) % 显示图像


% 创建yolo网络
inputSize = [224 224 3];
numClasses = width(vehicleDataset)-1; % 通过table的列数计算类别数

% 用于评估锚框个数
trainingDataForEstimation = transform(trainingData,@(data)preprocessData(data,inputSize));
numAnchors = 7;
[anchorBoxes, meanIoU] = estimateAnchorBoxes(trainingDataForEstimation, numAnchors)


% 特征提取层采用resnet50
featureExtractionNetwork = resnet50;

featureLayer = 'activation_40_relu';

% 设置yolo网络
lgraph = yolov2Layers(inputSize,numClasses,anchorBoxes,featureExtractionNetwork,featureLayer);

% 进行数据增强
augmentedTrainingData = transform(trainingData,@augmentData);

% 可视化增强后的图片
augmentedData = cell(4,1);
for k = 1:4
    data = read(augmentedTrainingData);
    augmentedData{k} = insertShape(data{1},'Rectangle',data{2});
    reset(augmentedTrainingData);
end
figure
montage(augmentedData,'BorderSize',10)

% 对增强数据进行预处理
preprocessedTrainingData = transform(augmentedTrainingData,@(data)preprocessData(data,inputSize));

data = read(preprocessedTrainingData);

% 显示一下
I = data{1};
bbox = data{2};
annotatedImage = insertShape(I,'Rectangle',bbox);
annotatedImage = imresize(annotatedImage,2);
figure
imshow(annotatedImage)


% 训练参数
options = trainingOptions('sgdm', ...
        'MiniBatchSize', 16, ....
        'InitialLearnRate',1e-3, ...
        'MaxEpochs',20,...
        'CheckpointPath', tempdir, ...
        'Shuffle','never');
    
if doTraining       
    % 训练YOLOv2检测器
    [detector,info] = trainYOLOv2ObjectDetector(preprocessedTrainingData,lgraph,options);
else
    % 载入预训练模型
    pretrained = load('yolov2_mytrain.mat');
    detector = pretrained.detector;
end


% 测试训练好的模型并显示
I = imread(testDataTbl.imageFilename{4});
I = imresize(I,inputSize(1:2));
[bboxes,scores] = detect(detector,I);

I = insertObjectAnnotation(I,'rectangle',bboxes,scores);
figure
imshow(I)

% 预处理测试集
preprocessedTestData = transform(testData,@(data)preprocessData(data,inputSize));
% 对测试集数据进行测试
detectionResults = detect(detector, preprocessedTestData);
% 评估准确率
[ap,recall,precision] = evaluateDetectionPrecision(detectionResults, preprocessedTestData);

figure
plot(recall,precision)
xlabel('Recall')
ylabel('Precision')
grid on
title(sprintf('Average Precision = %.2f',ap))



% 图像增强
function B = augmentData(A)
% 应用随机水平翻转和随机X/Y缩放图像；
% 如果重叠大于0.25，则在边界外缩放的框将被裁减；
% 变换图像颜色
B = cell(size(A));

I = A{1};
sz = size(I);
if numel(sz)==3 && sz(3) == 3
    I = jitterColorHSV(I,...
        'Contrast',0.2,...
        'Hue',0,...
        'Saturation',0.1,...
        'Brightness',0.2);
end

% 随机翻转和缩放图像
tform = randomAffine2d('XReflection',true,'Scale',[1 1.1]);
rout = affineOutputView(sz,tform,'BoundsStyle','CenterOutput');
B{1} = imwarp(I,tform,'OutputView',rout);

% 对锚框进行相同的变换
[B{2},indices] = bboxwarp(A{2},tform,rout,'OverlapThreshold',0.25);
B{3} = A{3}(indices);

% 当框的数据不存在时返回原始数据
if isempty(indices)
    B = A;
end
end
