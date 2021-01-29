function [F, Out] = sstv_lrtf1(Y,opts)

tol =1e-4;
maxit=80;
if isfield(opts,'eta');    eta=opts.eta;     end
if isfield(opts,'rank');    rank=opts.rank;    end
if isfield(opts, 'gamma');         gamma = opts.gamma;              end
if isfield(opts, 'beta');        beta = opts.beta;                end
if isfield(opts, 'lambda1');      lambda1 = opts.lambda1;                end
if isfield(opts, 'lambda2');     lambda2= opts.lambda2;                end
if isfield(opts, 'alpha');          alpha = opts.alpha;              end
max_beta=1e10;
max_gamma=1e10;

Nway = size(Y);
%% Initialization
P1 = zeros(Nway);
P2=P1;
P3=P1;
P4=P1;
P5=P1;

S=P1;
X=randn(Nway);%X=Y
F=P1;

h=Nway(1);w=Nway(2);d=Nway(3);
Eny_x   = ( abs(psf2otf([+1; -1], [h,w,d])) ).^2  ;
Eny_y   = ( abs(psf2otf([+1, -1], [h,w,d])) ).^2  ;
Eny_z   = ( abs(psf2otf([+1, -1], [w,d,h])) ).^2  ;
Eny_z   =  permute(Eny_z, [3, 1 2]);
determ  =  Eny_x + Eny_y + Eny_z;
% H =1;
% eigHtH      = abs(fftn(H, Nway)).^2;
% eigDtD      = abs(alpha(1)*fftn([1 -1],  Nway)).^2 + abs(alpha(2)*fftn([1 -1]', Nway)).^2;
% d_tmp(1,1,1)= 1; d_tmp(1,1,2)= -1;
% eigEtE  = abs(alpha(3)*fftn(d_tmp, Nway)).^2;
% Htg         = imfilter(g, H, 'circular');
[D,Dt]      = defDDt(alpha);

Out.ResT=[]; Out.PSNR=[];

for k = 1:maxit
    
    %% solve F-subproblem
    Q=0.5*(Y+X-S+(P1+P2)/beta);
    Q=permute(Q,[1,3,2]);
    F1= prox_tnn(Q,rank,1/(2*beta));
    F=ipermute(F1,[1,3,2]);
    %% solve A,B,C-subproblem
    [Df1 Df2 Df3] = D(X);
    A= prox_l1(Df1-P3/gamma,(lambda1*alpha(1))/gamma);
    B= prox_l1(Df2-P4/gamma,(lambda1*alpha(2))/gamma);
    C= prox_l1(Df3-P5/gamma,(lambda1*alpha(3))/gamma);
    
    %% solve X-subproblem
    temp =beta*(F-P1/beta);
    rhs   = fftn(temp +gamma*( Dt(A+P3/gamma,  B+P4/gamma, C+P5/gamma)));
    eigA  = beta+ gamma*determ;
    X     = real(ifftn(rhs./eigA));
    
    %% Solve S-subproblem
    S= prox_l1(Y-F+P2/beta,lambda2/beta);
    
    %% check the convergence
    if isfield(opts, 'Xtrue')
        FT=opts.Xtrue;
        resT=norm(F(:)-FT(:))/norm(FT(:));
        psnr=PSNR3D(F*255,FT*255);
        Out.ResT = [Out.ResT,resT];
        Out.PSNR = [Out.PSNR,psnr];
    end
    res1=norm(Y(:)-S(:)-F(:))/norm(F(:));
    res2=max(abs((F(:)-X(:))));
    
    if k==1 || mod(k, 10) == 0
        if isfield(opts, 'Xtrue')
            fprintf('SSTV_LRTF: iter = %d   PSNR=%f   res=%f  \n', k, psnr,  resT);
        else
            fprintf('SSTV_LRTF: iter = %d   \n', k);
        end
    end
    if res1 < tol && res2<tol
        break;
    end
    %% update Lagrange multiplier
    P1=P1+beta*(X-F);
    P2=P2+beta*(Y-F-S);
    
    P3=P3+gamma*(A-Df1);
    P4=P4+gamma*(B-Df2);
    P5=P5+gamma*(C-Df3);
    
    beta = min(eta* beta, max_beta);
    gamma = min(eta* gamma, max_gamma);
    
    imshow(F(:,:,77));
    drawnow;
end
end

function [D,Dt] = defDDt(beta)
D  = @(U) ForwardD(U, beta);
Dt = @(X,Y,Z) Dive(X,Y,Z, beta);
end

function [dfx,dfy,dfz] = ForwardD(tenX, weight)
sizeD=size(tenX);
dfx      = zeros(sizeD);
dfy      = zeros(sizeD);
dfz      = zeros(sizeD);
dfx(1:end-1,:,:) = diff(tenX, 1, 1);
dfx(end,:,:)     =  tenX(1,:,:) - tenX(end,:,:);
dfy(:,1:end-1,:) = diff(tenX, 1, 2);
dfy(:,end,:)     = tenX(:,1,:) - tenX(:,end,:);
dfz(:,:,1:end-1) =diff(tenX, 1, 3);
dfz(:,:,end)     = tenX(:,:,1) - tenX(:,:,end);
dfx=weight(1)*dfx;dfy=weight(2)*dfy;dfz=weight(3)*dfz;
end

function DtXYZ = Dive(tenX,tenY,tenZ, weight)
sizeD=size(tenX);
dfxT   = zeros(sizeD);
dfyT   = zeros(sizeD);
dfzT   = zeros(sizeD);
dfxT(1,:,:) = tenX(end, :, :) - tenX(1, :, :); %
dfxT(2:end,:,:) = -diff(tenX, 1, 1);
dfyT(:,1,:)     =  tenY(:,end,:) - tenY(:,1,:);
dfyT(:,2:end,:) = -diff(tenY, 1, 2);
dfzT(:,:,1)     = tenZ(:,:,end) - tenZ(:,:,1);
dfzT(:,:,2:end) = -diff(tenZ, 1, 3);

DtXYZ = weight(1)*dfxT + weight(2)*dfyT+weight(3)*dfzT ;
end
