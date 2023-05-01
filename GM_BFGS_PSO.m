function [y_est,fval] = GM_BFGS_PSO(Rx, Rg, dRg, epsilon, NPtcl, Nmax, omega_max, omega_min, c1, c2, S)
%GM_BFGS_PSO

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

    if (PSO_idx >= Nmax)
        break
    end

end

y_est = pg*scale;

    function obj = obj_fun(y_vec)

        obj = 0;
        
        for m = 1:M
            for l = 1:L
                obj = obj + (Rg(m,l)-norm(y_vec(1:H)-y_vec(H+(m-1)*H+1:H+(m-1)*H+H))-norm(y_vec(1:H)-Rx(:,l)))^2/((Rg(m,l)-norm(y_vec(1:H)-y_vec(H+(m-1)*H+1:H+(m-1)*H+H))-norm(y_vec(1:H)-Rx(:,l)))^2 + epsilon^2) ...
                    + (dRg(m,l)-norm(y_vec(H+(m-1)*H+1:H+(m-1)*H+H)-Rx(:,l)))^2/((dRg(m,l)-norm(y_vec(H+(m-1)*H+1:H+(m-1)*H+H)-Rx(:,l)))^2 + epsilon^2);
            end
        end

    end

end

