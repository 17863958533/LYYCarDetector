% 绘制训练曲线
% clear;
% clc

figure;
plot(info.TrainingLoss);
grid on 
xlabel('迭代次数');
ylabel('训练损失');

% 利用训练好的模型进行一次测试
img=imread('F:/毕业设计程序/南方测绘杯论文及图片/testData/白天车辆1.jpg');
%img=imread('‪E:/毕业设计程序/VehicleDetection/train/data/vehicleImages/image_00026.jpg');%更换为本地路径
[bboxes, scores]=detect(detector, img);

% 标记图像
if (~isempty(bboxes))
    img=insertObjectAnnotation(img, 'rectangle', bboxes, scores);
end
figure;
imshow(img);