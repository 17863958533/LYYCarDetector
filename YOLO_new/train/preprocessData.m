% ͼ��Ԥ����
function data = preprocessData(data,targetSize)
% ����ͼƬ��Bbox��С��targetSize
scale = targetSize(1:2)./size(data{1},[1 2]);
data{1} = imresize(data{1},targetSize(1:2));
% disp(data{2})
data{2} = bboxresize(data{2},scale);
end