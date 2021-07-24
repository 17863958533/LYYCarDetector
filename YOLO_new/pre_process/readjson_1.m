clear
clc

addpath('./jsonlab')
fname = './bdd100k_labels_images_val.json'; %待读取的文件名称
jsonData = loadjson(fname); % jsonData是个struct结构
