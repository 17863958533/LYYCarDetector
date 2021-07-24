%可以输入需要测试模型的绝对路径
I = imread('../train/data/vehicleImages/image_00015.jpg');
I = imresize(I,[224 224]);
[bboxes,scores] = detect(detector,I);%这里的detector，请确定在工作区的是哪一个方法的

I = insertObjectAnnotation(I,'rectangle',bboxes,scores);
figure
imshow(I)