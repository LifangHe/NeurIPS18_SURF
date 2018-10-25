function [cvLambda,cvTestErr,optW,resSD,optiLambda,SolutionPath,TestErrInterp,TestRSD,cvSigma] = main_SURF(Xten,Xt,Y);
%UNTITLED4 Summary of this function goes here
%%   Detailed explanation goes here
%% Inputs
%   Xten          Input data with I1 * I2 * ... * IN * M - multi-way features by samples
%   Xt            Input data with M * prod(In) - samples by features
%   Y             Input label with M * 1

%% Outputs
%   cvLambda      The solution path of K-fold in train set
%                 - a cell with R*1, each cell is a matrix with K * MaxIter
%   cvTestErr     The predicted error of K-fold in test set
%   optW             
%   resSD         The standard deviation of residual

% addpath('tensor_toolbox/');
addpath('tensorlab/');

%% Parameter Setting
% rng('default');               % For reproducibility
rng(1,'twister')
R = 50;                         % Rank of CP decomposition
MaxIter = 1000;                 % Length of solution path
early_stop = 1;
nstep_wait = 300;
alpha = 1;                      % L2-norm regularized parameter
M = size(Xten,ndims(Xten));     % Number of samples
dim = size(Xten);

epsilon = 0.1;
xi = epsilon*0.005;

%% Cross Validation
cvp = cvpartition(M,'Kfold',5);           % K-fold Cross Validation
% cvp = cvpartition(M,'HoldOut',0.2);     % HoldOut Validation

% ------------------------------------------------------
% Perform model fit for each rank
% ------------------------------------------------------
lastSD = 10000;
threshold = 1e-4;
optW = zeros(1,prod(dim(1:end-1)));
resSD = sqrt(mse(Y));
for r = 1:R
    fprintf('Current rank is: r=%g \n', r);
    [cvLambda{r,1}, cvTestErr{r,1},cvW,cvSigma] = cvfun(cvp, Xten, Xt, Y,epsilon,xi,MaxIter,alpha,early_stop,nstep_wait);
     [optiLambda(r),SolutionPath{r,1},TestErrInterp{r,1},TestRSD(r),absW] = MyInterp(cvLambda{r,1},cvTestErr{r,1},cvW,cvSigma);   
     [W, res] = MyTrain(Xten, Xt,Y,epsilon,xi,alpha,absW);
     if sum(W) ==0
         break;
     else
         Y = res;
         resSD(r) = sqrt(mse(Y));            % The Standard Deviation of Residual (divide by M, not M-1)
         fprintf('The residual standard deviation of rank-%g is %g \n', r, resSD(r));
         end
         if abs(resSD(r)-lastSD) < threshold
             fprintf('The residual standard deviation starts increase \n');
             break;
         else
             optW(r,:) = W;
             lastSD = resSD(r);
         end

end

%% ===================================================
%                cvfun() 
% ===================================================
function [Lambda, TestErr,TestW,TestSigma] = cvfun(cvp, Xten, Xt, Y,epsilon,xi,MaxIter,alpha,early_stop,nstep_wait)
    Lambda = cell(cvp.NumTestSets,1);
%     TrainErr = cell(cvp.NumTestSets,1);
    TestErr = cell(cvp.NumTestSets,1);
    TestW = cell(cvp.NumTestSets,1);
    TestSigma = cell(cvp.NumTestSets,1);
    for i = 1:cvp.NumTestSets
        trIdx = cvp.training(i);
        teIdx = cvp.test(i);
        if ndims(Xten)==3
            Xtrain = Xten(:,:,trIdx);
            Xtest  = Xten(:,:,teIdx);
        elseif ndims(Xten)==4
            Xtrain = Xten(:,:,:,trIdx);
            Xtest  = Xten(:,:,:,teIdx);
        end
        Ytrain = Y(trIdx);
        Ytest  = Y(teIdx);
        Xttrain = Xt(trIdx,:);
        [Lambda{i,1},TestErr{i,1},TestW{i,1},TestSigma{i,1}] = TrainAndPredict(Xtrain,Ytrain, ...
        Xtest,Ytest,Xttrain,epsilon,xi,MaxIter,alpha,early_stop,nstep_wait);    
    end
end

%% ===================================================
%                MyInterp() 
% ====================================================
function [optiLamb,SolPath,TestErrInt,TestErrRSD,absW] = MyInterp(Lambda,TestErr,tew,sigma)
    SolPath = cell2mat(Lambda');
    SolPath = unique(SolPath,'stable');
    for i = 1:length(Lambda)
        x = Lambda{i,1};
        y = TestErr{i,1};
        [xu,idx] = unique(x,'last');
        yu = y(idx); 
       %% Square Interpolation
       if length(xu)>=2
           errSqrt = interp1(xu,sqrt(yu),SolPath,'linear','extrap');
           TestErrInt(i,:) = errSqrt.^2;
           if sum(TestErrInt(i,:)<0)>=1
               fprintf('Interpolation error requires nonnegative value in each point.\n');
               break;
           end
       else
           TestErrInt(i,:) = ones(1,length(SolPath)) * yu;
       end
       %% Linear Interpolation
%        TestErrInt(i,:) = interp1(xu,yu,SolPath,'linear','extrap');
%        TrainErrInt(i,:) = interp1(xu,zu,SolPath,'linear','extrap');
    end
    errm = mean(TestErrInt,1);
    [errmin,Ind] = min(errm);
    TestErrRSD = sqrt(errmin);
    optiLamb = SolPath(Ind);
    fprintf('The average test error of K-fold is %g \n',TestErrRSD);
%% Compute the average L1 norm of W
    absW = 0;
    p = 0;
    for i=1:5
        idxx = find(optiLamb==Lambda{i,1});
        if ~isempty(idxx)
            wt = cellfun(@transpose,tew{i,1}(max(idxx),:),'un',0);
            wt = cpdgen(wt) * sigma{i,1}(max(idxx));
            absW = absW + sum(abs(wt(:)));
            p = p+1;
        end
    end
    absW = absW/p;
end
end
