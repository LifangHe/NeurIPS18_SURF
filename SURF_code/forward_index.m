function [n_hat,i_hat,s_hat,J_max] = forward_index(Ze, Zd, dim, epsilon, M)
% ===================================================
%                Forward Step
% ===================================================
%% Reuse the results from backward step
% Ze = 2 * e_hat' * Z_hat = 2 * (Z_n * y - Z_n * Z_n' * w_hat - alpha * beta_n * M * w_hat)'
% Zd = epsilon * Diag(Z_hat' * Z_hat) = epsilon * (Diag(Z_n' * Z_n) + alpha * beta_n * M)

%% 
N = length(dim);
n_hat = 0;
i_hat = 0;
s_hat = 0;
J_max = -1000000;
for n = 1: N
    Ze_n = Ze{n,1};
    value = abs(Ze_n) - Zd{n,1};
    [J_max_new, ind] = max(value);
    if J_max < J_max_new
        J_max = J_max_new;
        n_hat = n;
        i_hat = ind;
        s_hat = sign(Ze_n(i_hat)) * epsilon;
    end
end
J_max = J_max * epsilon / M;
end