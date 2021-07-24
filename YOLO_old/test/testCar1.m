% I = imread('../train/data/vehicleImages/image_00015.jpg');
I = imread('C:/Users/dell/Desktop/testData/image_00027.jpg');
I = imresize(I,[224 224]);
[bboxes,scores] = detect(detector,I);

I = insertObjectAnnotation(I,'rectangle',bboxes,scores);
figure
imshow(I)