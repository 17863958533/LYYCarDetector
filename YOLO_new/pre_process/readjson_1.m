clear
clc

addpath('./jsonlab')
fname = './bdd100k_labels_images_val.json'; %����ȡ���ļ�����
jsonData = loadjson(fname); % jsonData�Ǹ�struct�ṹ
