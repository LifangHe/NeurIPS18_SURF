function [lambda] =  forward_update_lambda(delta_J, lambda, epsilon, xi)
lambda_new = (delta_J - xi)/epsilon;
lambda = min(lambda,lambda_new);
end