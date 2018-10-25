function [lambda, test_err, w, sigma] = TrainAndPredict(Xtrain,Ytrain,Xtest,Ytest,Xttrain, epsilon,xi,MaxIter,alpha, early_stop, nstep_wait);
%UNTITLED2 Summary of this function goes here
%%   Detailed explanation goes here
%  Xtrain        Input training data in tensor form
%  Ytrain        Input training label with one column
%  Xtest         Input test data in tensor form
%  Ytest         Input test label with one column
%  Xttrain       Input training data in vector form - Matrix: M*d.

%% Outputs
%  lambda         Solution path
%  test_err       The mean square error of predicted results on test data 
%   W             Coefficient tensorS
%  sigma          sigma value 

%% Set defalt parameters
if nargin < 11
    fprintf('Not enough input! Terminate now.\n')
    return;
end
lambda_stop = 0;
%% Initialize
N        = ndims(Xtrain) - 1;         % Number of modes
M        = size(Xtrain,N+1);          % Number of training samples
sz       = size(Xtrain);
for i=1:N+1
    DIM{1,i} = 1:sz(i);
end
dim      = sz(1:N);
mode = cumsum(ones(1,N));
threshold = 1e-3;
t = 1;
%% Initialize solution by {w} = (s1, 1, ..., 1)
% Xttrain = Unfold(Xtrain, sz, N+1); % Data in vector form. Matrix: M*d.
XY = Xttrain' * Ytrain;
diag_XX = sum(Xttrain.^2, 1);
[J_max, ind_max] = max(2*abs(XY)-diag_XX'*epsilon);
if J_max~=0
    s = sign(XY(ind_max)) * epsilon;
else
    fprintf('J_max is zeros! Terminate now.\n')
    return;
end
[I{1:N}] = ind2sub(dim,ind_max);
for i=1:N
    w{t,i} = zeros(1,dim(i));
    w{t,i}(1,I{1,i}) = 1;
end
sigma(t) = epsilon;
w{t,1} = sign(s) * w{t,1};
lambda(t) = J_max/M - alpha * epsilon;

%% Calculate training and test error
inds = I;
inds{N+1} = 1:size(Xtest,N+1);
% Ytrain_Pred = squeeze(Xtrain(inds{:}))* sigma(t) * sign(s);
% train_err(t,1) = mse(Ytrain,Ytrain_Pred);
Ytest_Pred = squeeze(Xtest(inds{:}))* sigma(t) * sign(s);
test_err(t,1) = mse(Ytest, Ytest_Pred);

%% Calculate Z -- Z^{(-n)}
Z = cell(1,N);
for i=1:N
    inds_n = DIM;
    mode_n = setdiff(mode,i);
    inds_n(mode_n) = I(mode_n);
    if i==1
        Z{i} = squeeze(Xtrain(inds_n{:}));
    else
        Z{i} = squeeze(Xtrain(inds_n{:}))* sign(s);
    end
end
%% Main Steps
while t < MaxIter
    [m_hat, i_hat, s_hat, delta_Gamma, Ze, Zd,~] = backward_index(Z, Ytrain, sigma(t), w(t,:), dim, I, alpha, epsilon, mode);
    if m_hat~= 0 && -delta_Gamma <= lambda(t) * epsilon - xi
        fprintf('The %g -th backward optimamum is: m_hat=%g, i_hat = %g, s_hat=%g \n', t, m_hat, i_hat,s_hat);
        w_hat = w{t,m_hat} * sigma(t);
        [sigma_update, w_update, I] = backward_update(w_hat, m_hat, i_hat, s_hat, I);
        if sigma_update==0
            fprintf('sigma equals to zero! Terminate now.\n')
            break;
        else
            sigma(t+1) = sigma_update;
            w(t+1,:) = w(t,:);
            w{t+1,m_hat} = w_update;
            lambda(t+1) = lambda(t);
        end
    else
        [m_hat,i_hat,s_hat, delta_J] = forward_index(Ze, Zd, dim, epsilon, M);
        fprintf('The %g -th forward is: m_hat=%g, i_hat = %g, s_hat=%g \n', t, m_hat, i_hat, s_hat);
        if m_hat~=0
            w_hat = w{t,m_hat} * sigma(t);
            [sigma_update, w_update, I] = forward_update_w(w_hat, m_hat, i_hat, s_hat, I);
            if sigma_update==0
                fprintf('Sigma equals to zero! Terminate now.\n')
                break;
            else
                sigma(t+1) = sigma_update;
                w(t+1,:) = w(t,:);
                w{t+1,m_hat} = w_update;
                lambda(t+1) =  forward_update_lambda(delta_J, lambda(t), epsilon, xi);
            end
            if lambda(t+1) < 0
                fprintf('Lambda =%g! Terminate now.\n',lambda(t+1));
                lambda(t+1) = [];
                w(t+1,:) = [];
                sigma(t+1) = [];
                break;
            end
        else          
            fprintf('No any further index is selected! Terminate now.\n')
            break;
        end
    end
 %% Update Z_n
    inds_n = DIM;
    inds_n{m_hat} = i_hat;
    Xtrain_m = Xtrain(inds_n{:});
    if N==2
        j = setdiff(mode,m_hat);
        INC = squeeze(Xtrain_m) * s_hat;
        Z{j} = sigma(t) * Z{j} + INC;
        Z{j} = Z{j}/sigma(t+1);         
%          Ytrain_Pred = Ytrain_Pred + INC' * w{t+1,j};
%          train_err(t+1,1) = mse(Ytrain,Ytrain_Pred);
    else
        for i = 1:N
            if i~=m_hat
                mode_n = setdiff(mode,[i,m_hat]);
                tmp = tmprod(Xtrain_m,w(t+1,mode_n),mode_n);   
                INC = squeeze(tmp) * s_hat;
                Z{i} = sigma(t) * Z{i} + INC;
                Z{i} = Z{i}/sigma(t+1);
            end
        end
    end
%%  Calculate test error
    inds_n{N+1} = 1:size(Xtest,N+1);
    Xtest_m = Xtest(inds_n{:});
    if N==2
        j = setdiff(mode,m_hat);
        INC = squeeze(Xtest_m) * s_hat;
        Ytest_Pred = Ytest_Pred + INC' * w{t+1,j}';
        test_err(t+1,1) = mse(Ytest, Ytest_Pred);
    else
        mode_n = setdiff(mode,m_hat);
        tmp = tmprod(Xtest_m,w(t+1,mode_n),mode_n);
        INC = squeeze(tmp) * s_hat;
        Ytest_Pred = Ytest_Pred + INC';
        test_err(t+1,1) = mse(Ytest, Ytest_Pred);
    end    
%%  Check Terminzation
    start = t - nstep_wait + 1;
    if early_stop==1 && start > 0
        osc = test_err(start) - test_err(t+1);
        if osc < threshold
            lambda_stop = 1;
        end
        if lambda_stop ==1 && lambda(t+1)-lambda(t) < 0
%             train_err(t+1) = [];
            test_err(t+1) = [];
            lambda(t+1) = [];
            w(t+1,:) = [];
            sigma(t+1) = [];
            fprintf('The %g -th iteration terminates \n', t);
            break;
        end
    end
    t = t+1;
end
end
