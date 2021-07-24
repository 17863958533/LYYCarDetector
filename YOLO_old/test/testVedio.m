clear
clc


load('yolov2_lyy_smalldata_trained_model.mat','detector')

inputSize = [448 448 3];

v = VideoReader('test.avi'); %测试组1
%v = VideoReader('mytestData2.mp4'); %测试组2

writer = VideoWriter('transcoded_video_changed.avi'); %测试组1
%writer = VideoWriter('mytestResultData.avi'); %测试组2

open(writer);
while hasFrame(v)
    video = readFrame(v);
    
    I = video;
    I = imresize(I,inputSize(1:2));
    
     [bboxes,scores] = detect(detector,I);

    if scores
        I = insertObjectAnnotation(I,'rectangle',bboxes,scores);
    end
    
    I = imresize(I, size(video,[1,2]));
    imshow(I)
    writeVideo(writer,I);
    
end
close(writer);