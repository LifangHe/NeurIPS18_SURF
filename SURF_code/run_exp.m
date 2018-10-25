function SURF = run_exp();
clc;

addpath('tensorlab/');

data_name = 'SimData';
method_name = 'SURF';

%% Initialize
data         = load(data_name);
Nt           = size(data.Xt,2);
trainInd     = data.trainInd;
testInd      = data.testInd;
t            = 5;                   % 5-fold cross validation
rng('default');


%% Run methods to match different input format
X = data.Xt;
Y = data.Y;
% Xten = data.Xten;
for i = 1:t
    fprintf('The %d th repeat \n',i);
    if ndims(data.Xten)==3
        Xtrain = data.Xten(:,:,trainInd{i,1});
    elseif ndims(data.Xten)==4
        Xtrain = data.Xten(:,:,:,trainInd{i,1});
    end
    time1 = cputime;
    [~,~,Wt,~,~,~,~,~] = main_SURF(Xtrain, X(trainInd{i,1},:),Y(trainInd{i,1})); 
    Xtest    = X(testInd{i,1},:);
    Ytest = Xtest * Wt';
    Ytest = sum(Ytest,2);
    time(i) = cputime - time1;
    RMSE(i) = sqrt(mse(Y(testInd{i,1}), Ytest));
    S(i)    = sum(sum(Wt,1)==0)/Nt;
    W{i}    = Wt;
end
SURF = struct();
SURF.RMSE = RMSE;
SURF.S = S;
SURF.W = W;
SURF.time = time;
filename = strcat('Result/',method_name,'_',data_name);
save([filename '.mat'],'SURF');
result = [mean(RMSE), mean(S), mean(time); ...
          std(RMSE), std(S), std(time)];
csvwrite([filename '.csv'], result);
       
end
% end
