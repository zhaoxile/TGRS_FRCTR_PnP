function [L] = rsvd(A,rank)
[n1,n2,n3]=size(A);
Ak = fft(A,[],3);
parfor k= 1:n3
    Bi=Ak(:,:,k);
    [U,S,V]=svd(Bi,'econ');
    U1=U(:,1:rank);
    S1=S(1:rank,1:rank);
    V1=V(:,1:rank);
    L(:,:,k)=U1*S1*V1';
end
L=ifft(L,[],3);
