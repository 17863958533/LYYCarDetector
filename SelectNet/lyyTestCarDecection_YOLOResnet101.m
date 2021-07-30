clear all
clc

%%
% 判断是否开始训练
% True:是，不使用官方的模型
% False:否，使用官方的模型
doTraining = true;
% if ~doTraining && ~exist('yolov2ResNet50VehicleExample_19b.mat','file')    
%     disp('Downloading pretrained detector (98 MB)...');
%     pretrainedURL = 'https://www.mathworks.com/supportfiles/vision/data/yolov2ResNet50VehicleExample_19b.mat';
%     websave('yolov2ResNet50VehicleExample_19b.mat',pretrainedURL);
% end
%%
%%
% 载入训练数据集，这里使用BDD100K车辆数据集
%大训练样本量载入（大小样本二选一）
% data = load('./others/data/carDatasetGroundTruth.mat');
% vehicleDataset = data.carDataset; % table型，包含文件路径和groundTruth
%小训练样本量载入（大小样本二选一）
 data = load('./others/data/vehicleDatasetGroundTruth.mat');
 vehicleDataset = data.vehicleDataset; % table型，包含文件路径和groundTruth

%%
%%
% 添加绝对路径至vehicleDataset中
vehicleDataset.imageFilename = fullfile([pwd, '/others/data/'],vehicleDataset.imageFilename);
% 显示数据集中的一个图像，以了解它包含的图像的类型。
vehicleDataset(1:4,:) % 显示部分数据情况
%%
%%
% 将数据集分成两部分：一个是用于训练检测器的训练集，一个是用于评估检测器的测试集。
% 选择 60% 的数据进行训练，10%的数据用于测试，30%的数据用于评估训练后的检测器。
%值得注意的是，这部分划分是可以进行比例调整的，后续需要根据不同的学习率进行试验测试对比！！！
rng(0); %控制随机数的生成
shuffledIndices = randperm(height(vehicleDataset));
idx = floor(0.6 * length(shuffledIndices) );

trainingIdx = 1:idx;
trainingDataTbl = vehicleDataset(shuffledIndices(trainingIdx),:);

validationIdx = idx+1 : idx + 1 + floor(0.1 * length(shuffledIndices) );
validationDataTbl = vehicleDataset(shuffledIndices(validationIdx),:);

testIdx = validationIdx(end)+1 : length(shuffledIndices);
testDataTbl = vehicleDataset(shuffledIndices(testIdx),:);
%%
%%
% 保存数据和标签
imdsTrain = imageDatastore(trainingDataTbl{:,'imageFilename'});
bldsTrain = boxLabelDatastore(trainingDataTbl(:,'vehicle'));

imdsValidation = imageDatastore(validationDataTbl{:,'imageFilename'});
bldsValidation = boxLabelDatastore(validationDataTbl(:,'vehicle'));

imdsTest = imageDatastore(testDataTbl{:,'imageFilename'});
bldsTest = boxLabelDatastore(testDataTbl(:,'vehicle'));

% 整理训练测试集
trainingData = combine(imdsTrain,bldsTrain);
validationData = combine(imdsValidation,bldsValidation);
testData = combine(imdsTest,bldsTest);
%%
%%
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
% 创建yolo网络,这里注意要根据提取网络来选择设置inputSize
inputSize = [224 224 3];
numClasses = width(vehicleDataset)-1; % 通过table的列数计算类别数
% 用于评估锚框个数
trainingDataForEstimation = transform(trainingData,@(data)preprocessData(data,inputSize));
numAnchors = 7;
[anchorBoxes, meanIoU] = estimateAnchorBoxes(trainingDataForEstimation, numAnchors)
% 特征提取层采用resnet18
featureExtractionNetwork = resnet101;

featureLayer = 'res5a_relu';
%%
%%
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
preprocessedValidationData = transform(validationData,@(data)preprocessData(data,inputSize));

data = read(preprocessedTrainingData);

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
options = trainingOptions('sgdm', ...
        'MiniBatchSize',16, ....
        'InitialLearnRate',1e-3, ...
        'MaxEpochs',20, ... 
        'CheckpointPath',tempdir, ...
        'ValidationData',preprocessedValidationData);
 
%判断是否开始训练
if doTraining       
    % Train the YOLO v2 detector.
    [detector,info] = trainYOLOv2ObjectDetector(preprocessedTrainingData,lgraph,options);
else
    % Load pretrained detector for the example.
    pretrained = load('yolov2ResNet50VehicleExample_19b.mat');
    detector = pretrained.detector;
end
%%
%%
% 测试训练好的模型并显示
I = imread(testDataTbl.imageFilename{4});
I = imresize(I,inputSize(1:2));
[bboxes,scores] = detect(detector,I);

I = insertObjectAnnotation(I,'rectangle',bboxes,scores);
figure
imshow(I)
%%
%%
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
%%
%%
%官方的支持函数
function B = augmentData(A)
% Apply random horizontal flipping, and random X/Y scaling. Boxes that get
% scaled outside the bounds are clipped if the overlap is above 0.25. Also,
% jitter image color.

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

% Randomly flip and scale image.
tform = randomAffine2d('XReflection',true,'Scale',[1 1.1]);
rout = affineOutputView(sz,tform,'BoundsStyle','CenterOutput');
B{1} = imwarp(I,tform,'OutputView',rout);

% Sanitize box data, if needed.
%A{2} = helperSanitizeBoxes(A{2}, sz);

% Apply same transform to boxes.
[B{2},indices] = bboxwarp(A{2},tform,rout,'OverlapThreshold',0.25);
B{3} = A{3}(indices);

% Return original data only when all boxes are removed by warping.
if isempty(indices)
    B = A;
end
end

% 图像预处理
function data = preprocessData(data,targetSize)
% 调整图片和Bbox大小至targetSize
scale = targetSize(1:2)./size(data{1},[1 2]);
data{1} = imresize(data{1},targetSize(1:2));
% disp(data{2})
data{2} = bboxresize(data{2},scale);
end
%%
%%

