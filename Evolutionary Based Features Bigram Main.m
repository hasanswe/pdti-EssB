clc;
clear all;
close all;

BigramPSSM_feature=[];
for i=1:1858     
    i
    arr=[];
    newDataPSSM=csvread(['pssm' num2str(i),'.xls']);
    arr(:,:)=newDataPSSM(:,1:20);
    BigramPSSM_feature=[BigramPSSM_feature; Bigram_PSSM(arr)];
end
clearvars -except BigramPSSM_feature
save BigramPSSM_feature BigramPSSM_feature;
