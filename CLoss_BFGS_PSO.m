function [y_est,fval] = CLoss_BFGS_PSO(Rx, Rg, dRg, sigma, NPtcl, Nmax, omega_max, omega_min, c1, c2, S)
%CLoss_BFGS_PSO

[M, ~] = size(Rg);
[H, L] = size(Rx);

fval = [];

scale = 1000;

Rx = Rx/scale;
Rg = Rg/scale;
dRg = dRg/scale;

options = optimoptions(@fminunc,'Display','off','Algorithm','quasi-newton');
    
pjk_mtx = 2*S*(rand((M+1)*H, NPtcl) - 0.5)/scale;

pjb_mtx = pjk_mtx;

obj_vec = zeros(NPtcl, 1);

for n_idx = 1:NPtcl
    obj_vec(n_idx) = obj_fun(pjb_mtx(:,n_idx));
end

min_idx = find(obj_vec == min(obj_vec));

pg = pjb_mtx(:,min_idx);

PSO_idx = 0;

vjkp1_mtx = zeros((M+1)*H, NPtcl);

while 1

    pj_tilde_mtx = zeros((M+1)*H, NPtcl);

    for n_idx = 1:NPtcl

        [y_est_int,~,~,~] = fminunc(@obj_fun,pjk_mtx(:,n_idx),options);

        pj_tilde_mtx(:, n_idx) = y_est_int;

        obj_value_pjb = obj_fun(pjb_mtx(:,n_idx));

        if ((obj_fun(pj_tilde_mtx(:, n_idx)))<obj_value_pjb)
            pjb_mtx(:,n_idx) = pj_tilde_mtx(:, n_idx);
        end

    end

    for n_idx = 1:NPtcl
        obj_vec(n_idx) = obj_fun(pjb_mtx(:,n_idx));
    end

    min_idx = find(obj_vec == min(obj_vec));

    pg = pjb_mtx(:,min_idx);

    fval = [fval;obj_fun(pg)];

    omega_k = omega_max - ((omega_max - omega_min)/Nmax)*PSO_idx;

    vjkp1_mtx = omega_k*vjkp1_mtx + c1*rand*(pjb_mtx - pjk_mtx) + c2*rand*(pg*ones(1,NPtcl) - pjk_mtx);

    pjkp1_mtx = pjk_mtx + vjkp1_mtx;

    while (sum(abs(pjkp1_mtx)>=S)~=0)

        vjkp1_mtx = omega_k*vjkp1_mtx + c1*rand*(pjb_mtx - pjk_mtx) + c2*rand*(pg*ones(1,NPtcl) - pjk_mtx);

        pjkp1_mtx = pjk_mtx + vjkp1_mtx;

    end

    PSO_idx = PSO_idx + 1;

    pjk_mtx = pjkp1_mtx;

    %update sigma
    sigma_old = sigma;
    Error = [];
    for m_idx = 1:M
        for l_idx = 1:L
            Error = [Error; (Rg(m_idx,l_idx)-norm(pg(1:H)-pg(H+(m_idx-1)*H+1:H+(m_idx-1)*H+H))-norm(pg(1:H)-Rx(:,l_idx)))];
        end
    end
    for m_idx = 1:M
        for l_idx = 1:L
            Error = [Error; (dRg(m_idx,l_idx)-norm(pg(H+(m_idx-1)*H+1:H+(m_idx-1)*H+H)-Rx(:,l_idx)))];
        end
    end
    sgm_E = std(Error);
    R = iqr(Error);
    
    sigma = max(1.06*min(sgm_E, R/1.34)*(2*M*L)^(-0.2),sigma_old);

    if (PSO_idx >= Nmax)
        break
    end

end

y_est = pg*scale;

    function obj = obj_fun(y_vec)

        obj = 0;
        
        for m = 1:M
            for l = 1:L
                obj = obj + 1 - exp(-(Rg(m,l)-norm(y_vec(1:H)-y_vec(H+(m-1)*H+1:H+(m-1)*H+H))-norm(y_vec(1:H)-Rx(:,l)))^2/(2*sigma^2)) ...
                    + 1 - exp(-(dRg(m,l)-norm(y_vec(H+(m-1)*H+1:H+(m-1)*H+H)-Rx(:,l)))^2/(2*sigma^2));
            end
        end

    end

end

