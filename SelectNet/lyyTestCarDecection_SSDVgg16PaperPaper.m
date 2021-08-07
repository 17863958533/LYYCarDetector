clear all;
clc;

%%
%确定是否开始训练
doTraining = true;
% if ~doTraining && ~exist('ssdResNet50VehicleExample_20a.mat','file')
%     disp('Downloading pretrained detector (44 MB)...');
%     pretrainedURL = 'https://www.mathworks.com/supportfiles/vision/data/ssdResNet50VehicleExample_20a.mat';
%     websave('ssdResNet50VehicleExample_20a.mat',pretrainedURL);
% end
%%
%%
% 载入训练数据集，这里使用BDD100K车辆数据集
%大训练样本量载入（大小样本二选一）
% data = load('./others/data//carDatasetGroundTruth.mat');
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
% 选择 60% 的数据进行训练，10%的数据用于测试，20%的数据用于评估训练后的检测器。
%值得注意的是，这部分划分是可以进行比例调整的，后续需要根据不同的学习率进行试验测试对比！！！
rng(0);
shuffledIndices = randperm(height(vehicleDataset));
idx = floor(0.6 * length(shuffledIndices) );
trainingData = vehicleDataset(shuffledIndices(1:idx),:);
testData = vehicleDataset(shuffledIndices(idx+1:end),:);

% 保存数据和标签
imdsTrain = imageDatastore(trainingData{:,'imageFilename'});
bldsTrain = boxLabelDatastore(trainingData(:,'vehicle'));

imdsTest = imageDatastore(testData{:,'imageFilename'});
bldsTest = boxLabelDatastore(testData(:,'vehicle'));

% 整理训练测试集
trainingData = combine(imdsTrain,bldsTrain);
testData = combine(imdsTest, bldsTest);

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
%创建SSD网络
inputSize = [300 300 3];
numClasses = width(vehicleDataset)-1;
lgraph = ssdLayers(inputSize, numClasses, 'vgg16');

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
preprocessedTrainingData = transform(augmentedTrainingData,@(data)preprocessData(data,inputSize));
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
        'MiniBatchSize', 16, ....
        'InitialLearnRate',1e-1, ...
        'LearnRateSchedule', 'piecewise', ...
        'LearnRateDropPeriod', 30, ...
        'LearnRateDropFactor', 0.8, ...
        'MaxEpochs', 300, ...
        'VerboseFrequency', 50, ...        
        'CheckpointPath', tempdir, ...
        'Shuffle','every-epoch');

%确定是否训练
if doTraining
    % Train the SSD detector.
    [detector, info] = trainSSDObjectDetector(preprocessedTrainingData,lgraph,options);
else
    % Load pretrained detector for the example.
    pretrained = load('ssdResNet50VehicleExample_20a.mat');
    detector = pretrained.detector;
end
%%
%%
% 测试训练好的模型并显示
data = read(testData);
I = data{1,1};
I = imresize(I,inputSize(1:2));
[bboxes,scores] = detect(detector,I, 'Threshold', 0.4);

I = insertObjectAnnotation(I,'rectangle',bboxes,scores);
figure
imshow(I)
%%
%%
% 预处理测试集
preprocessedTestData = transform(testData,@(data)preprocessData(data,inputSize));
% 对测试集数据进行测试
detectionResults = detect(detector, preprocessedTestData, 'Threshold', 0.4);
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
%官方支持函数
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

% Sanitize boxes, if needed.
%A{2} = helperSanitizeBoxes(A{2}, sz);
    
% Apply same transform to boxes.
[B{2},indices] = bboxwarp(A{2},tform,rout,'OverlapThreshold',0.25);    
B{3} = A{3}(indices);
    
% Return original data only when all boxes are removed by warping.
if isempty(indices)
    B = A;
end
end

function data = preprocessData(data,targetSize)
% Resize image and bounding boxes to the targetSize.
sz = size(data{1},[1 2]);
scale = targetSize(1:2)./sz;
data{1} = imresize(data{1},targetSize(1:2));

% Sanitize boxes, if needed.
%data{2} = helperSanitizeBoxes(data{2}, sz);

% Resize boxes.
data{2} = bboxresize(data{2},scale);
end
%%









