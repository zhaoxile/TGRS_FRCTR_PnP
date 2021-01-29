function [X,S,Out] = FRCTR_BM3D(Y,opts)

%% initial value of parameters
Nway = size(Y);

tol      = 1e-3;
max_iter = 80;


lambda1  = opts.lambda1;
rho=opts.rho;
beta = opts.beta;
max_beta = 1e10;
rank=opts.rank;
%% initialization

L =Y;
X=Y;
S = zeros(Nway);
F1 = S;
F2 = S;
F3 = S;
P1 = S;
P2 = S;
P3 = S;
P4 = S;
P5 = S;

Out.Res=[]; Out.PSNR=[];
for iter = 1 : max_iter
    %% Let
    Xold = X;
    X1 = permute(X,[2,3,1]);  X2 = permute(X,[3,1,2]);  X3 = X;
    p1 = permute(P1,[2,3,1]); p2 = permute(P2,[3,1,2]); p3 = P3;
   %% update F   
%     F1=rsvd(X1+p1/beta(1),rank(1));
%     F2=rsvd(X2+p2/beta(2),rank(2));
%     F3=rsvd(X3+p3/beta(3),rank(3));
     F1=rtsvd(X1+p1/beta(1),rank(1),30);
     F2=rtsvd(X2+p2/beta(2),rank(2),30);
     F3=rtsvd(X3+p3/beta(3),rank(3),50);
     F1 = ipermute(F1,[2,3,1]);
     F2 = ipermute(F2,[3,1,2]);
    %% update L
    temp =X+P5/beta(5);
    parfor i=1:Nway(3)
        maxt(i)=max(max(temp(:,:,i)));mint(i)=min(min(temp(:,:,i)));
        temp(:,:,i)=(temp(:,:,i)-mint(i))/(maxt(i)-mint(i));
        [~,t]=BM3D(1,temp(:,:,i),sqrt(lambda1/beta(5))); 
        L(:,:,i)=(maxt(i)-mint(i))*t+mint(i);
    end
    
    %% update S
    S = prox_l1(Y-X+P4/beta(4),1/beta(4));
    
    %% update X
   temp = beta(1)*F1-P1 + beta(2)*F2-P2 + beta(3)*F3-P3 + beta(4)*(Y-S)+P4+beta(5)*L-P5;
    X = temp/sum(beta);
    
    %% check the convergence
    chg=norm(Xold(:)-X(:))/norm(Xold(:));
    Out.Res = [Out.Res,chg];
    if isfield(opts, 'Xtrue')
        XT=opts.Xtrue;
        psnr = PSNR3D(XT * 255, L * 255);
        Out.PSNR = [Out.PSNR,psnr];
    end
    
    if iter==1 || mod(iter, 10) == 0
        if isfield(opts, 'Xtrue')
       fprintf('FRCTR_BM3D: iter = %d   PSNR= %f   res= %f \n', iter, psnr, chg);
        else
            fprintf('FRCTR_BM3D: iter = %d   res= %f \n', iter, chg);
        end
    end
    
    if chg < tol
        break;
    end
    
    %% update M & P
    
    P1 = P1 + beta(1)*(X-F1);
    P2 = P2 +beta(2)*(X-F2);
    P3 = P3 + beta(3)*(X-F3);
    P4=P4+beta(4)*(Y-X-S);
    P5=P5+beta(5)*(X-L);
    beta = min(rho.*beta,max_beta);
    
    imshow(L(:,:,45));
    drawnow;
end