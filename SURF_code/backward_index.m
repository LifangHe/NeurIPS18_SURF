function [n_hat, i_hat, s_hat, Gamma_max, Ze, Zd, e] = backward_index(Z, y, sigma, w, dim, I, alpha, epsilon,mode)
%% ===================================================
%                Backforward Step
% ====================================================
N = length(dim);
M = length(y);
n_hat = 0;
i_hat = 0;
s_hat = 0;
Gamma_max = -1000000;
e = y - Z{1}' * w{1}' * sigma;     % e is constant at each iteration
Ze = cell(N,1);                   % Ze = 2 * e_hat' * Z_hat = 2 * (Z_n * (y - Z_n' * w_hat) - alpha * beta_n * M * w_hat)'
Zd = cell(N,1);                   % Zd = epsilon * Diag(Z_hat' * Z_hat) = epsilon * (Diag(Z_n' * Z_n) + alpha * beta_n * M)
for n = 1: N
    I_n = I{n};
    mode_n = setdiff(mode,n);
    w_hat = w{n} * sigma;
    s = -sign(w_hat);
    Z_n = Z{n};
%     Z_n = tmprod(Xten,w(mode_n),mode_n);
%     Z_n = squeeze(Z_n);
    beta_n = cellfun(@norm,w(mode_n));
    beta_n = prod(beta_n)^2;
    Mab = M * alpha * beta_n;
    diag_ZZ = sum(Z_n.^2, 2);
    Ze{n,1} = (Z_n * e - Mab * w_hat') * 2;
    Zd{n,1} = (diag_ZZ + Mab) * epsilon;
%     ZZ_n = Z_n * Z_n';
%     Ze{n,1} = (Z_n * y - ZZ_n * w_hat' - Mab * w_hat') * 2;
%     Zd{n,1} = (diag(ZZ_n) + Mab) * epsilon;
    object = Ze{n,1} .* s' - Zd{n,1};
    value = object(I_n);
    [Gamma_max_new, ind] = max(value);
    if Gamma_max < Gamma_max_new
        Gamma_max = Gamma_max_new;
        n_hat = n;
        i_hat = I_n(ind);
        s_hat = s(i_hat) * epsilon;
    end
end
Gamma_max = Gamma_max * epsilon / M;
end