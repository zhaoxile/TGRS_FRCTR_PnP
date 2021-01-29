function [L,S,Out] = FRCTR(X,opts)
%% initial value of parameters
Nway = size(X);

tol      = 1e-3;
max_iter = 80;
rho      =opts.rho;
mu       = opts.mu;
beta     = opts.beta;
rank     = opts.rank;
max_mu   =1e10*[1,1,1];
max_beta = 1e10;

%% initialization

L = zeros(Nway);
S = L;
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
   %% update Z
     Z1=rtsvd(L1+m1/mu(1),rank(1),30);
     Z2=rtsvd(L2+m2/mu(2),rank(2),30);
     Z3=rtsvd(L3+m3/mu(3),rank(3),50);
     Z1 = ipermute(Z1,[2,3,1]);
     Z2 = ipermute(Z2,[3,1,2]);
    %% update L
    temp = mu(1)*(Z1-M1/mu(1)) + mu(2)*(Z2-M2/mu(2)) + mu(3)*(Z3-M3/mu(3)) + beta*(X-S+P/beta);
    L = temp/(beta+sum(mu));
    
    %% update S
    S = prox_l1(X-L+P/beta,1/beta);
    
    %% check the convergence
    dM = X-L-S;
    chg=norm(Lold(:)-L(:))/norm(Lold(:));
    Out.Res = [Out.Res,chg];
    if isfield(opts, 'Xtrue')
        XT=opts.Xtrue;
        psnr = PSNR3D(XT * 255, L * 255);
        Out.PSNR = [Out.PSNR,psnr];
    end
    
    if iter==1 || mod(iter, 10) == 0
        if isfield(opts, 'Xtrue')
            fprintf('FRCTR: iter = %d   PSNR= %f   res= %f \n', iter, psnr, chg);
        else
            fprintf('FRCTR: iter = %d   res= %f \n', iter, chg);
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
    
    imshow(L(:,:,71));
    drawnow;
end