clear
clc

% load('yolov2_bdd100k_8val.mat','detector')
% load('yolov2_mytrain.mat', 'detector')
load('yolov2_lyy_bigdata_trained_model.mat', 'detector')

% load('fasterRCNN_mathwork.mat', 'detector')


inputSize = [448 448 3];

% �����ļ�ѡ���ѡ��һ��ͼƬ
[file,path] = uigetfile({'*.jpg;*.jpeg;*.png;*.bmp;*.tif',...
    'ͼƬ�ļ� (*.jpg,*.jpeg,*.png,*.bmp,*.tif)'},'ѡ��һ��ͼƬ');
fileName= fullfile(path,file); % ѡ���ͼƬ����·��

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
