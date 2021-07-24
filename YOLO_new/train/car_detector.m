clear
clc

doTraining = true; % �Ƿ����ѵ��

% ��ѹ����
% ��һ����8 9���е������Ǵ�ѵ���������ݣ����ڱ�����ʹ�õ���Сѵ����������,�������ֶ�ѡһ
% data = load('./data/carDatasetGroundTruth.mat');
% vehicleDataset = data.carDataset; % table�ͣ������ļ�·����groundTruth
data = load('./VehicleDetection/train/data/vehicleDatasetGroundTruth.mat');
vehicleDataset = data.vehicleDataset; % table�ͣ������ļ�·����groundTruth

% ��Ӿ���·����vehicleDataset��
vehicleDataset.imageFilename = fullfile([pwd, '/VehicleDetection/train/data/'],vehicleDataset.imageFilename);

% ��ʾ���ݼ��е�һ��ͼ�����˽���������ͼ������͡�
vehicleDataset(1:4,:) % ��ʾ�����������

% �����ݼ��ֳ������֣�һ��������ѵ���������ѵ������һ������������������Ĳ��Լ���
% ѡ�� 70% �����ݽ���ѵ����������������������
rng(0); % �������������
shuffledIndices = randperm(height(vehicleDataset));
idx = floor(0.7 * length(shuffledIndices) );
trainingDataTbl = vehicleDataset(shuffledIndices(1:idx),:);
testDataTbl = vehicleDataset(shuffledIndices(idx+1:end),:);

% �������ݺͱ�ǩ
imdsTrain = imageDatastore(trainingDataTbl{:,'imageFilename'}); % ·��
bldsTrain = boxLabelDatastore(trainingDataTbl(:,'vehicle')); % ��ʵ������

imdsTest = imageDatastore(testDataTbl{:,'imageFilename'});
bldsTest = boxLabelDatastore(testDataTbl(:,'vehicle'));

% ����ѵ�����Լ�
trainingData = combine(imdsTrain,bldsTrain); % �����ļ�·������ʵ��
testData = combine(imdsTest,bldsTest);


% ��ʾ����
data = read(trainingData); % data����ͼƬ���ݡ���ʵ�����ꡢ���
I = data{1};
bbox = data{2};
annotatedImage = insertShape(I,'Rectangle',bbox); % �����ݾ����б����ʵ��
annotatedImage = imresize(annotatedImage,2);
figure
imshow(annotatedImage) % ��ʾͼ��


% ����yolo����
inputSize = [224 224 3];
numClasses = width(vehicleDataset)-1; % ͨ��table���������������

% ��������ê�����
trainingDataForEstimation = transform(trainingData,@(data)preprocessData(data,inputSize));
numAnchors = 7;
[anchorBoxes, meanIoU] = estimateAnchorBoxes(trainingDataForEstimation, numAnchors)


% ������ȡ�����resnet50
featureExtractionNetwork = resnet50;

featureLayer = 'activation_40_relu';

% ����yolo����
lgraph = yolov2Layers(inputSize,numClasses,anchorBoxes,featureExtractionNetwork,featureLayer);

% ����������ǿ
augmentedTrainingData = transform(trainingData,@augmentData);

% ���ӻ���ǿ���ͼƬ
augmentedData = cell(4,1);
for k = 1:4
    data = read(augmentedTrainingData);
    augmentedData{k} = insertShape(data{1},'Rectangle',data{2});
    reset(augmentedTrainingData);
end
figure
montage(augmentedData,'BorderSize',10)

% ����ǿ���ݽ���Ԥ����
preprocessedTrainingData = transform(augmentedTrainingData,@(data)preprocessData(data,inputSize));

data = read(preprocessedTrainingData);

% ��ʾһ��
I = data{1};
bbox = data{2};
annotatedImage = insertShape(I,'Rectangle',bbox);
annotatedImage = imresize(annotatedImage,2);
figure
imshow(annotatedImage)


% ѵ������
options = trainingOptions('sgdm', ...
        'MiniBatchSize', 16, ....
        'InitialLearnRate',1e-3, ...
        'MaxEpochs',20,...
        'CheckpointPath', tempdir, ...
        'Shuffle','never');
    
if doTraining       
    % ѵ��YOLOv2�����
    [detector,info] = trainYOLOv2ObjectDetector(preprocessedTrainingData,lgraph,options);
else
    % ����Ԥѵ��ģ��
    pretrained = load('yolov2_mytrain.mat');
    detector = pretrained.detector;
end


% ����ѵ���õ�ģ�Ͳ���ʾ
I = imread(testDataTbl.imageFilename{4});
I = imresize(I,inputSize(1:2));
[bboxes,scores] = detect(detector,I);

I = insertObjectAnnotation(I,'rectangle',bboxes,scores);
figure
imshow(I)

% Ԥ������Լ�
preprocessedTestData = transform(testData,@(data)preprocessData(data,inputSize));
% �Բ��Լ����ݽ��в���
detectionResults = detect(detector, preprocessedTestData);
% ����׼ȷ��
[ap,recall,precision] = evaluateDetectionPrecision(detectionResults, preprocessedTestData);

figure
plot(recall,precision)
xlabel('Recall')
ylabel('Precision')
grid on
title(sprintf('Average Precision = %.2f',ap))



% ͼ����ǿ
function B = augmentData(A)
% Ӧ�����ˮƽ��ת�����X/Y����ͼ��
% ����ص�����0.25�����ڱ߽������ŵĿ򽫱��ü���
% �任ͼ����ɫ
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

% �����ת������ͼ��
tform = randomAffine2d('XReflection',true,'Scale',[1 1.1]);
rout = affineOutputView(sz,tform,'BoundsStyle','CenterOutput');
B{1} = imwarp(I,tform,'OutputView',rout);

% ��ê�������ͬ�ı任
[B{2},indices] = bboxwarp(A{2},tform,rout,'OverlapThreshold',0.25);
B{3} = A{3}(indices);

% ��������ݲ�����ʱ����ԭʼ����
if isempty(indices)
    B = A;
end
end
