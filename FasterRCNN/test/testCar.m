clear
clc

% load('yolov2_bdd100k_8val.mat','detector')
% load('RCNN_lyy_smalldata_trained_model.mat', 'detector')
load('RCNN_lyy_bigdata_trained_model.mat', 'detector')

% load('fasterRCNN_mathwork.mat', 'detector')


inputSize = [448 448 3];

% 弹出文件选择框，选择一张图片
[file,path] = uigetfile({'*.jpg;*.jpeg;*.png;*.bmp;*.tif',...
    '图片文件 (*.jpg,*.jpeg,*.png,*.bmp,*.tif)'},'选择一张图片');
fileName= fullfile(path,file); % 选择的图片绝对路径

if file
    Im = imread(fileName);
%   I = imresize(Im,inputSize(1:2));
    I = Im;
    [bboxes,scores] = detect(detector,I);

    if scores
        I = insertObjectAnnotation(I,'rectangle',bboxes,round(scores,2), 'FontSize',8);
    end
    I = imresize(I,size(Im,[1, 2]));
    imshow(I)
end


