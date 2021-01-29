function [L] = rtsvd(A,rank,p)
[n1,n2,n3]=size(A);
% W=zeros(n2,(rank+p),n3);
% W(:,:,1)=randn(n2,(rank+p));
Ak = fft(A,[],3);
% Wk = fft(W,[],3);
Wi=randn(n2,(rank+p));
parfor k= 1:n3
    Ai = Ak(:,:,k);
    %matrix size
%     Wi=Wk(:,:,k);
    Yi=Ai*Wi;
    [Q,R]=qr(Yi,0);
%     for i=1:q
%         Zi=Ai'*Q;
%         [q,~]=qr(Zi,0);
%         Yi=Ai*q;
%         [Q,~]=qr(Yi,0);
%     end
    Q=Q(:,1:rank+p);
    Bi=Q'*Ai;
    [U,S,V]=svd(Bi,'econ');
    S=S(1:rank,1:rank);
    V=V(:,1:rank);
    U=U(:,1:rank);
    U=Q*U;
    
    L(:,:,k)=U*S*V';
end
L=ifft(L,[],3);
