% This is a demo for HWLRR
clc
clear
addpath('Functions/')
addpath('Datasets/')
dataset_name = 'ORL_32x32';
lambda1 = 0.1;
lambda2 = 0.1;
k = 2;
load(strcat(dataset_name,'.mat'))
fea = fea';
fea = fea./repmat(sqrt(sum(fea.^2)),[size(fea,1) 1]);
n = length(gnd);
nnClass = length(unique(gnd));
[Z obj dis] = HWLRR(fea, lambda1, lambda2, k+1, 99);
addpath('Ncut_9');
Z_out = Z;
A = Z_out;
A = A - diag(diag(A));
A = abs(A);
A = (A+A')/2;  
[NcutDiscrete,NcutEigenvectors,NcutEigenvalues] = ncutW(A,nnClass);
result_label = zeros(size(fea,2),1);
for f = 1:nnClass
    id = find(NcutDiscrete(:,f));
    result_label(id) = f;
end
result = ClusteringMeasure(gnd, result_label);
acc  = result(1)
nmi  = result(2)       