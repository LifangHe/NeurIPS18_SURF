function [sigma_update, w_update, I_update] = backward_update(w_hat, m_hat, i_hat, s_hat, I)
I_update = I;
if abs(w_hat(i_hat)) > abs(s_hat)
    w_hat(i_hat) = w_hat(i_hat) + s_hat;
else
    w_hat(i_hat) = 0;
end
%% L1 norm normalization
sigma_update = sum(abs(w_hat));
if sigma_update~=0
    w_update = w_hat/sigma_update;
    if w_update(i_hat)==0
        I_update{1,m_hat} = setdiff(I{1,m_hat}, i_hat);
    end
else
    w_update = 0;
    return;
end
end