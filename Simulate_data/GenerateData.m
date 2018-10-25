function GenerateData();
%UNTITLED3 Summary of this function goes here
%   Detailed explanation goes here
clear; close all; clc;

% % Fix stream of random numbers
s1 = RandStream.create('mrg32k3a','Seed',1);
s0 = RandStream.setGlobalStream(s1);

filename = 'SimData.mat';

sp = 0.8;         % Sparity rate
nd = 16;
dim = [16,16];
d = prod(dim);
n = 500;

%% Creat correlation matrix
correlate = zeros(d,d);
for i=1:d
    for j=1:d
        if i<=j
            [r1,c1] = ind2sub(dim,i);
            [r2,c2] = ind2sub(dim,j);
            p = sqrt((r1-r2)^2 + (c1-c2)^2);
            correlate(i,j) = 0.6^p;
        else
            correlate(i,j) = correlate(j,i);
        end
    end
end
 
%% Simulate data
mu = zeros(d,1);
Sigma = correlate; 
C = chol(Sigma);
Z = repmat(mu',n,1) + randn(n,d)*C;

% mu = zeros(d,1);
% Z = mvnrnd(mu, correlate, n);S

%% Simulate coefficients
R = 50;
W = 0;
for i=1:R
    w1 = randn(dim(1),1);
    w1 = w1/sum(abs(w1));
    w2 = randn(dim(2),1);
    w2 = w2/sum(abs(w2));
    W = W + 1/i * w1 * w2';
end
density = 1-sp;
S = sprand(dim(1),dim(2),density);
S(S>0)=1;
S = full(S);
W = W.*S;
W = full(W);

% Z = bsxfun(@times,Z,S(:)');

%% Generate data with different dimensions
% ndim = nd;
% ind = 1:d;
% A = reshape(ind,dim);
% idx = A(1:ndim,1:ndim);
% Xt = Z(:,idx);
% Wt = W(1:ndim,1:ndim);

ndim = nd;
Xt = Z;
Wt = W;

%% Preprocess data
Xt = normalize(Xt);

%% Simulate responses
yOriginal = Xt*Wt(:);
amplitude = 1;  % noise level
Y = yOriginal + amplitude*randn(n,1);
% Y = center(Y);

%% Tensor Format
parfor i = 1:n
    Xten(:,:,i) = reshape(Xt(i,:),ndim,ndim);
end

% Restore random stream
RandStream.setGlobalStream(s0);

%% Divide data for 5-fold cross validation
for i = 1:5
    [trInd,valInd,testInd{i,1}] = dividerand(n,4,1,1);
    trainInd{i,1} = cat(1, trInd',valInd');
end
save(filename,'Xt','Xten','Y','trainInd','testInd');
end
