function [L,S,N,Out] = de_3DTNN(X,opts)

% Solve the 3DLogTNN-based HSI mixed noise removal model by ADMM
% Created by Yu-Bang Zheng ��zhengyubang@163.com��
% 8/20/2019

%% initial value of parameters
Nway = size(X);

tol      = 1e-3;
max_iter = 80;
rho      =1.2;

alpha   =[1,1,opts.theta]/(2+opts.theta);
lambda1 = opts.phi/opts.sigma;
% lambda1 = opts.phi;
lambda11 = 1/sqrt(max(Nway(3),Nway(2))*Nway(1));
lambda12 = 1/sqrt(max(Nway(1),Nway(3))*Nway(2));
lambda13 = 1/sqrt(max(Nway(1),Nway(2))*Nway(3));
lambda   = alpha(1)*lambda11+alpha(2)*lambda12+alpha(3)*lambda13;
lambda2  = opts.varpi*lambda;

tau      = opts.omega*[1,1,1];
mu       = alpha./tau;
beta     = 1/mean(tau);

max_mu   =1e10*[1,1,1];
max_beta = 1e10;

%% initialization

L = zeros(Nway);
S = L;
N = L;
Z1 = L;
Z2 = L;
Z3 = L;
M1 = L;
M2 = L;
M3 = L;
P =  L;

Out.Res=[]; Out.PSNR=[];
for iter = 1 : max_iter
    %% Let
    Lold = L;
    L1 = permute(L,[2,3,1]);  L2 = permute(L,[3,1,2]);  L3 = L;
    m1 = permute(M1,[2,3,1]); m2 = permute(M2,[3,1,2]); m3 = M3;
    %     Z1=permute(Z1,[2,3,1]); Z2 = permute(Z2,[3,1,2]);Z3=Z3;
%     %% update Z
    tau = alpha./mu;
    Z1 = ipermute(prox_tnn_my(L1+m1/mu(1),tau(1)),[2,3,1]);
    Z2 = ipermute(prox_tnn_my(L2+m2/mu(2),tau(2)),[3,1,2]);
    Z3 = prox_tnn_my(L3+m3/mu(3),tau(3));
 
%     %
    %% update L
    temp = mu(1)*(Z1-M1/mu(1)) + mu(2)*(Z2-M2/mu(2)) + mu(3)*(Z3-M3/mu(3)) + beta*(X-N-S+P/beta);
    L = temp/(beta+sum(mu));
    
    %% update S
    S = prox_l1(X-N-L+P/beta,lambda2/beta);
    
    %% update N
    N = (beta*(X-L-S+P/beta))/(2*lambda1+beta);
    
    %% check the convergence
    dM = X-L-S-N;
    chg=norm(Lold(:)-L(:))/norm(Lold(:));
    Out.Res = [Out.Res,chg];
    if isfield(opts, 'Xtrue')
        XT=opts.Xtrue;
        psnr = PSNR3D(XT * 255, L * 255);
        Out.PSNR = [Out.PSNR,psnr];
    end
    
    if iter==1 || mod(iter, 10) == 0
        if isfield(opts, 'Xtrue')
            fprintf('3DTNN: iter = %d   PSNR= %f   res= %f \n', iter, psnr, chg);
        else
            fprintf('3DTNN: iter = %d   res= %f \n', iter, chg);
        end
    end
    
    if chg < tol
        break;
    end
    
    %% update M & P
    P = P + beta*dM;
    M1 = M1 + mu(1)*(L-Z1);
    M2 = M2 + mu(2)*(L-Z2);
    M3 = M3 + mu(3)*(L-Z3);
    beta = min(rho*beta,max_beta);
    mu = min(rho*mu,max_mu);
    
    imshow(L(:,:,77));
    drawnow;
end