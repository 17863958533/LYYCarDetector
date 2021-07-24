% 图像预处理
function data = preprocessData(data,targetSize)
% 调整图片和Bbox大小至targetSize
scale = targetSize(1:2)./size(data{1},[1 2]);
data{1} = imresize(data{1},targetSize(1:2));
% disp(data{2})
data{2} = bboxresize(data{2},scale);
end