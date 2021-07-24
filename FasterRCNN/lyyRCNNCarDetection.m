clear all;
clc;

%%
%确定是否开始训练
doTraining = true;
% if ~doTraining && ~exist('fasterRCNNResNet50EndToEndVehicleExample.mat','file')
%     disp('Downloading pretrained detector (118 MB)...');
%     pretrainedURL = 'https://www.mathworks.com/supportfiles/vision/data/fasterRCNNResNet50EndToEndVehicleExample.mat';
%     websave('fasterRCNNResNet50EndToEndVehicleExample.mat',pretrainedURL);
% end
%%
%%
% 载入训练数据集，这里使用BDD100K车辆数据集
%大训练样本量载入（大小样本二选一）
data = load('./VehicleDetection/train/data/carDatasetGroundTruth.mat');
vehicleDataset = data.carDataset; % table型，包含文件路径和groundTruth
%小训练样本量载入（大小样本二选一）
% data = load('./data/vehicleDatasetGroundTruth.mat');
% vehicleDataset = data.vehicleDataset; % table型，包含文件路径和groundTruth
%%
%%
% 添加绝对路径至vehicleDataset中
vehicleDataset.imageFilename = fullfile([pwd, '/VehicleDetection/train/data/'],vehicleDataset.imageFilename);
% 显示数据集中的一个图像，以了解它包含的图像的类型。
vehicleDataset(1:4,:) % 显示部分数据情况
%%
%%
% 将数据集分成两部分：一个是用于训练检测器的训练集，一个是用于评估检测器的测试集。
% 选择 60% 的数据进行训练，10%的数据用于测试，20%的数据用于评估训练后的检测器。
%值得注意的是，这部分划分是可以进行比例调整的，后续需要根据不同的学习率进行试验测试对比！！！
rng(0)
shuffledIndices = randperm(height(vehicleDataset));
idx = floor(0.6 * height(vehicleDataset)); %这里是数据划分的比例

trainingIdx = 1:idx;
trainingDataTbl = vehicleDataset(shuffledIndices(trainingIdx),:);

validationIdx = idx+1 : idx + 1 + floor(0.1 * length(shuffledIndices) );
validationDataTbl = vehicleDataset(shuffledIndices(validationIdx),:);

testIdx = validationIdx(end)+1 : length(shuffledIndices);
testDataTbl = vehicleDataset(shuffledIndices(testIdx),:);

% 保存数据和标签
imdsTrain = imageDatastore(trainingDataTbl{:,'imageFilename'});
bldsTrain = boxLabelDatastore(trainingDataTbl(:,'vehicle'));

imdsValidation = imageDatastore(validationDataTbl{:,'imageFilename'});
bldsValidation = boxLabelDatastore(validationDataTbl(:,'vehicle'));

imdsTest = imageDatastore(testDataTbl{:,'imageFilename'});
bldsTest = boxLabelDatastore(testDataTbl(:,'vehicle'));

%组合数据和标签
trainingData = combine(imdsTrain,bldsTrain);
validationData = combine(imdsValidation,bldsValidation);
testData = combine(imdsTest,bldsTest);

% 显示数据
data = read(trainingData);
I = data{1};
bbox = data{2};
annotatedImage = insertShape(I,'Rectangle',bbox);
annotatedImage = imresize(annotatedImage,2);
figure
imshow(annotatedImage)
%%
%%
%创建RCNN网络
inputSize = [224 224 3];
preprocessedTrainingData = transform(trainingData, @(data)preprocessData(data,inputSize));
numAnchors = 3;
anchorBoxes = estimateAnchorBoxes(preprocessedTrainingData,numAnchors)

%选择resnet50作为检测网络
featureExtractionNetwork = resnet50;
featureLayer = 'activation_40_relu';

numClasses = width(vehicleDataset)-1;

lgraph = fasterRCNNLayers(inputSize,numClasses,anchorBoxes,featureExtractionNetwork,featureLayer);
%%
%%
%数据增强
augmentedTrainingData = transform(trainingData,@augmentData);

%可视化增强后的图片
augmentedData = cell(4,1);
for k = 1:4
    data = read(augmentedTrainingData);
    augmentedData{k} = insertShape(data{1},'Rectangle',data{2});
    reset(augmentedTrainingData);
end

figure
montage(augmentedData,'BorderSize',10)
%%
%%
% 对增强数据进行预处理
trainingData = transform(augmentedTrainingData,@(data)preprocessData(data,inputSize));
validationData = transform(validationData,@(data)preprocessData(data,inputSize));

data = read(trainingData);

%显示一下
I = data{1};
bbox = data{2};
annotatedImage = insertShape(I,'Rectangle',bbox);
annotatedImage = imresize(annotatedImage,2);
figure
imshow(annotatedImage)
%%
%%
%训练参数设置
options = trainingOptions('sgdm',...
    'MaxEpochs',10,...
    'MiniBatchSize',2,...
    'InitialLearnRate',1e-3,...
    'CheckpointPath',tempdir,...
    'ValidationData',validationData);

%确定是否训练
if doTraining
    % Train the Faster R-CNN detector.
    % * Adjust NegativeOverlapRange and PositiveOverlapRange to ensure
    %   that training samples tightly overlap with ground truth.
    [detector, info] = trainFasterRCNNObjectDetector(trainingData,lgraph,options, ...
        'NegativeOverlapRange',[0 0.3], ...
        'PositiveOverlapRange',[0.6 1]);
else
    % Load pretrained detector for the example.
    pretrained = load('fasterRCNNResNet50EndToEndVehicleExample.mat');
    detector = pretrained.detector;
end
%%
%%
% 测试训练好的模型并显示
I = imread(testDataTbl.imageFilename{3});
I = imresize(I,inputSize(1:2));
[bboxes,scores] = detect(detector,I);

I = insertObjectAnnotation(I,'rectangle',bboxes,scores);
figure
imshow(I)
%%
%%
% 预处理测试集
testData = transform(testData,@(data)preprocessData(data,inputSize));
% 对测试集数据进行测试
detectionResults = detect(detector,testData,'MinibatchSize',4);   
% 评估准确率
[ap, recall, precision] = evaluateDetectionPrecision(detectionResults,testData);

figure
plot(recall,precision)
xlabel('Recall')
ylabel('Precision')
grid on
title(sprintf('Average Precision = %.2f', ap))
%%
%%
%官方支持函数
function data = augmentData(data)
% Randomly flip images and bounding boxes horizontally.
tform = randomAffine2d('XReflection',true);
sz = size(data{1});
rout = affineOutputView(sz,tform);
data{1} = imwarp(data{1},tform,'OutputView',rout);

% Sanitize box data, if needed.
%data{2} = helperSanitizeBoxes(data{2}, sz);

% Warp boxes.
data{2} = bboxwarp(data{2},tform,rout);
end

function data = preprocessData(data,targetSize)
% Resize image and bounding boxes to targetSize.
sz = size(data{1},[1 2]);
scale = targetSize(1:2)./sz;
data{1} = imresize(data{1},targetSize(1:2));

% Sanitize box data, if needed.
%data{2} = helperSanitizeBoxes(data{2}, sz);

% Resize boxes.
data{2} = bboxresize(data{2},scale);
end
%%
%%














