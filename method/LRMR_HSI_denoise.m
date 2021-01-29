function [ output_image ] = LRMR_HSI_denoise( oriData3_noise,r,M,s,stepszie )
% This package contains a matlab implementation of the LRMR  algorithm [1].
%
% Pleae run the "Simulated_demo.m" file,
% which will run our LRMR algorithm on the simulated indian data [2].
%
%
%[1]H. Zhang, W. He, L. Zhang, H. Shen, and  Q. Yuan, ��Hyperspectral Image 
%   Restoration Using Low-Rank Matrix Recovery,��
%   IEEE Trans. Geosci. Remote Sens. , vol. 52, pp. 4729-4743, Aug. 2014.
%[2] W. He, H. Zhang, L. Zhang, and  H. Shen, ��Total-Variation-Regularized 
%    Low-Rank Matrix Factorization for Hyperspectral Image Restoration,�� 
%    IEEE Trans. Geosci. Remote Sens. , vol. 54, pp. 178-188, Jan. 2016.
%
% NOTE:  LRMR is implemented with ssGoDec, which is different from the one 
%        mentioned in the original paper [1]. The selection of parameter s is 
%        changed correspondingly.

%
% Author: Wei He (November, 2014)
% E-mail addresses:(weihe1990@whu.edu.cn)
%% �ɵ��ڲ���
%  oriData3_noise                            noisy 3-D image normalized to [0,1] band by band
%  r (recommended value 7)                   the rank of each sub-cub(matrix version)
%  M (recommended value20)                   the spatial size of each sub cube (line*column=M*M)
%  s (recommended value 7000 for GoDec)      the number of pixels which are corrupted by sparse noise
%  s (recommended value q% for ssGoDec)      q% is the percentage of pixels which are corrupted by sparse noise
%  stepszie (recommended value 4)            stepsize of each sub-cub
%%%%%%%%%%%%%%%%  
 [m,n,p] = size(oriData3_noise);
 clear_image=zeros(m,n,p);
 
%  Sparse=zeros(m,n,p);            %ϡ����   
  
 Weight = zeros(m,n,p);      % weight matrix 
 patch_block = zeros(M^2,p); % the size of each block
%  sparse_block=zeros(M^2,p);          %ϡ����   
R         =   m-M+1;
C         =   n-M+1;
rr        =   [1:stepszie:R];
rr        =   [rr rr(end)+1:R];
cc        =   [1:stepszie:C];
cc        =   [cc cc(end)+1:C];
row       =   length(rr);
column    =   length(cc);


for   rownumber =1:row
     for columnnumber = 1:column
         i = rr(rownumber);
         j = cc(columnnumber); 
        for  k=1:1:p                     
         patch_reference = oriData3_noise(i:i+M-1,j:j+M-1,k); 
         patch_block(:,k) =  patch_reference(:);
         Weight(i:1:i+M-1,j:1:j+M-1,k) = Weight(i:1:i+M-1,j:1:j+M-1,k)+1;               
        end
%        [clear_patch_block,S,~,~]=GoDec( patch_block,r,s,0);   % used in the original paper
         [clear_patch_block,S,~,~]=SSGoDec( patch_block,r,s,0); % 
        for m2=1:1:p
          clear_image(i:1:i+M-1,j:1:j+M-1,m2) = reshape(clear_patch_block(:,m2),M,M)+clear_image(i:1:i+M-1,j:1:j+M-1,m2);
%         Sparse(i:1:i+M-1,j:1:j+M-1,m2) = reshape(S(:,m2),M,M)+Sparse(i:1:i+M-1,j:1:j+M-1,m2);                             %ϡ����
        end 
     end
            
end  
 Weight_last = 1./Weight;
 output_image = Weight_last.*clear_image;
end

%%
function[L,S,RMSE,error]=SSGoDec(X,rank,card,power)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                      Semi-Soft GoDec Algotithm
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%INPUTS:
%X: nxp data matrix with n samples and p features
%rank: rank(L)<=rank
%card: card(S)<=card
%power: >=0, power scheme modification, increasing it lead to better
%accuracy and more time cost
%OUTPUTS:
%L:Low-rank part
%S:Sparse part
%RMSE: error
%error: ||X-L-S||/||X||
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%REFERENCE:
%Tianyi Zhou and Dacheng Tao, "GoDec: Randomized Lo-rank & Sparse Matrix
%Decomposition in Noisy Case", ICML 2011
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Tianyi Zhou, 2011, All rights reserved.

%iteration parameters
iter_max=1e+2;
error_bound=1e-10;
iter=1;
RMSE=[];

%matrix size
[m,n]=size(X);
if m<n X=X'; end

%initialization of L and S
L=X;
S=sparse(zeros(size(X)));
          
while true

    %Update of L
    Y2=randn(n,rank);
    for i=1:power+1
        Y1=L*Y2;
        Y2=L'*Y1;
    end
    [Q,R]=qr(Y2,0);
    L_new=(L*Q)*Q';
    
    %Update of S
    T=L-L_new+S;
    L=L_new;
    S=wthresh(T,'s',card);
%     [~,idx]=sort(abs(T(:)),'descend');
%     S=zeros(size(X));S(idx(1:card))=T(idx(1:card));
    
    %Error, stopping criteria
    %T(idx(1:card))=0;
    T=T-S;
    RMSE=[RMSE norm(T(:))];
    if RMSE(end)<error_bound || iter>iter_max
        break;
    else
        L=L+T;
    end
    iter=iter+1;
    
end

LS=L+S;
error=norm(LS(:)-X(:))/norm(X(:));
if m<n 
    LS=LS';
    L=L';
    S=S';
end
end