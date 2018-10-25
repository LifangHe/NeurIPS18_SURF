function [sigma_update, w_update, I] = forward_update_w(w_hat, m_hat, i_hat, s_hat, I)
%% L1 norm normalization
w_hat(i_hat) = w_hat(i_hat) + s_hat;
sigma_update = sum(abs(w_hat));
if sigma_update~=0
    w_update = w_hat/sigma_update;
    I{1,m_hat} = union(I{1,m_hat},i_hat);
else
    w_update = 0;
    return;
end
end