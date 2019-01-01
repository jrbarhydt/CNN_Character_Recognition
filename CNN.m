% Neural Network Handwriting analysis
% Johnathon R Barhydt

%
% This code builds training and test sets from the MNIST database to
% classify handwriten digits 0-9 using both a single layer linear NN and a
% multiplayer CNN
%
clear all, close all, clc
%% 1-Layer NN
% number of trials
perms = 20;
% grab training/test data and combine for cross-validation
data=csvread('train.csv',1,0).';
% cross-validation of NN over several iterations
for i=1:perms
    %scramble data for cross validation
    perm=randperm(size(data,2)); 
    data_mixed=data(:,perm);
    %grab training inputs and labels
    X=data_mixed(2:end,1:30000);
    Y=data_mixed(1,1:30000);
    % solve for A
    A=Y*pinv(X);
    % test 1-layer NN
    % grab test inputs and labels
    X_test=data_mixed(2:end,30001:end).';
    Y_test=data_mixed(1,30001:end).';
    % solve for Y
    Y_predict=round(A*X_test.');
    % accuracy check
    Acc=(Y_predict==Y_test.');
    accuracy(i)=nnz(Acc)/size(Acc,2);

end
%% plot accuracy results
% cross validation result statistics
u_accuracy=round(100*mean(accuracy),1);
s_accuracy=round(100*std(accuracy),2); 
result = strcat(num2str(u_accuracy)," +/- ",num2str(s_accuracy),"% (Accuracy to 1 Std. Dev.)");
figure(1)
plot(accuracy)
set(gca,'Ylim',[0 1])
title('Accuracy of Single-Layer Neural Net')
ylabel('Ratio of Correct Predictions')
xlabel('Iteration Number')
legend(result,'Location','southeast')

%% multilayer NN
% 7 layer burrito! the recipe (order) is up to you
%-------------------------------------------------------------
%img is 28 by 28, grayscale, reshape training and test sets
X_sq=reshape(X,[28 28 1 30000]);
X_test_sq=reshape(X_test.',[28 28 1 12000]);
Y=categorical(Y);
Y_test=categorical(Y_test);
layers = [imageInputLayer([28 28 1]);
%important! only take a little window of the photo, slide the window across
%(looks for edges) perform bunch of functions to each window
convolution2dLayer(7,64); 
reluLayer();
maxPooling2dLayer(2,'Stride',2);

convolution2dLayer(4,64); 
reluLayer();
maxPooling2dLayer(2,'Stride',2);

fullyConnectedLayer(64);
fullyConnectedLayer(10);
softmaxLayer();
classificationLayer()];
options = trainingOptions('sgdm',...%stochastic gradient descent method 
    'Plots','training-progress',...%plot results
    'ValidationData',{X_test_sq,Y_test},...
    'ValidationFrequency',500,...
    'InitialLearnRate',0.0001); %start slow
%(batch gradient descent will grab more than one sample for each update)
rng('default') % For reproducibility
net = trainNetwork(X_sq,Y,layers,options); %data labels layers options

Y_predict = classify(net,X_test_sq);
accuracy = sum(Y_predict == categorical(Y_test))/numel(Y_test)


