clc; clear;

%% dataset path
train_data_path = '******   Your training dataset path\train\Training   ******';
test_data_path = '******   Your testing dataset path\test\Testing   ******';

%% feature extraction
fea_extract_train(train_data_path);
fea_extract_test(test_data_path)

%% training and testing
cla_train_test;