%% =================================================================
% This script performs two tensor fibered-rank-constrained optimization
% models for hyperspectral image restoration.
% listed as follows:
%     1. Fibered-rank-constrained tensor restoration (FRCTR) for HSI mixed
%        noise removal.
%     2. An FRCTR framework with an embedded Plug-and-Play (PnP)-based regularization
%        (FRCTR-PnP) for HSI mixed noise removal.
%
% More detail can be found in [1]
% [1] Yun-Yang Liu, Xi-Le Zhao*, Yu-Bang Zheng, Tian-Hui Ma, and Hongyan Zhang.
% Hyperspectral Image Restoration by Tensor Fibered Rank Constrained Optimization
% and Plug-and-Play Regularization
%
% Created by Yun-Yang Liu
% 8/20/2020

clear;
clc;
close all;

addpath(genpath('lib'));
addpath(genpath('data'));
addpath(genpath('method'));
addpath(genpath('BM3D-master'));

%% load noisy image
load('Pavia_G0.1.mat'); 
%% load clean image
load('cleanPavia.mat');   
Ohsi=img_clean; 


if max(Ohsi(:))>1
    Ohsi=my_normalized(Ohsi);
end

EN_FRCTR    =1;
EN_FRCTR_PnP=1;
methodname  = { 'Noise','FRCTR','FRCTR_PnP'};
Mnum = length(methodname);

%%
Nway = size(Ohsi);
%% evaluation indexes
Re_hsi  =  cell(Mnum,1);
psnr    =  zeros(Mnum,1);
ssim    =  zeros(Mnum,1);
sam     =  zeros(Mnum,1);
time    =  zeros(Mnum,1);
%%  corrupted image
i  = 1;
Re_hsi{i} = Nhsi;
[psnr(i), ssim(i), sam(i)] = MSIQA(Ohsi * 255, Re_hsi{i} * 255);
%% Performing FRCTR
i = i+1;
if EN_FRCTR
    %%%%%
    for mu1=[1e-3]
        for mu2=[1e-2]
            for bet=[mu1]
                for rho=[1.3]
                    opts=[];
                    opts.mu=[mu1,mu1,mu2]; %weighted coefficient of mode-k  fibered rank approximation
                    opts.rank=[4,4,110];   %fibered rank
                    opts.beta =bet;     %penalty parameters 
                    opts.rho=rho;
                    opts.Xtrue = Ohsi;
                    t0= tic;
                    [Re_hsi{i},~,Out]=FRCTR(Nhsi,opts);
                    time(i) = toc(t0)
                    [psnr(i), ssim(i), sam(i)] = MSIQA(Ohsi * 255, Re_hsi{i}* 255);
                    fprintf('FRCTR: PSNR=%5.4f   \n',  psnr(i));
                    imname=['mu1_',num2str(mu1),'_mu2_',num2str(mu2),'_beta_',num2str(bet),'_rho_',num2str(rho),'_result_psnr_',num2str(psnr(i)),'_ssim_',num2str(ssim(i)),'_sam_',num2str(sam(i)),'_time_',num2str(time(i)),'.mat'];
                    save(imname,'Re_hsi')
                end
            end
        end
    end
end
%% Performing FRCTR_PnP
i = i+1;
if EN_FRCTR_PnP
    opts=[];
    for lam1=[250]
        for rho1=[1.5]
            for beta1=[0.1]%0.15,0.1,0.05
                for beta3=[1e-3]%1e-2,1e-3,1e-4,1e-5
                    opts.rank=[4,4,110]; %fibered rank
                    opts.lambda1= lam1;  %tuning parameter
                    opts.rho=rho1*[1,1,1,1,1];
                    opts.beta =[beta1,beta1,beta3,beta1,beta1];  %penalty parameters
                    opts.Xtrue = Ohsi;
                    %%%%%
                    t0= tic;
                    [Re_hsi{i},~,Out]=FRCTR_BM3D(Nhsi,opts);
                    time(i) = toc(t0)
                    [psnr(i), ssim(i), sam(i)] = MSIQA(Ohsi * 255, Re_hsi{i}* 255);
                    fprintf('FRCTR_BM3D: PSNR=%5.4f   \n',  psnr(i));
                    imname=['lambda1_',num2str(lam1),'_rho1_',num2str(rho1),'_beta1_',num2str(beta1),'_beta3_',num2str(beta3),'_result_psnr_',num2str(psnr(i)),'_ssim_',num2str(ssim(i)),'_sam_',num2str(sam(i)),'_time_',num2str(time(i)),'.mat'];
                    save(imname,'Re_hsi')
                end
            end
        end
    end
end







