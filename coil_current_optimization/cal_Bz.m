clc;
clear all;

load('coil_trace.mat');
load('B0_map_valid.mat');

N_rec = 7;
N_tra = 8;

M=zeros(29,200,200,40);

data_coil_trace = cell(29,1);
cnt=1;
for i = 1:2
    if i == 1
        N = N_rec;
    else
        N = N_rec - 1;
    end 
    for j = 1:N
%         M(cnt,:,:,:)=cal_Bz_using_Biotsavart_28_june(X,Y,Z,20,ch29_coil_array_sub_FOV1.rec_coil(i,j).trace);
        data_coil_trace{cnt,1} = ch29_coil_array_sub_FOV1.rec_coil(i,j).trace;
        cnt=cnt+1;
    end
end
for i = 1:2
    for j = 1:N_tra
%         M(cnt,:,:,:)=cal_Bz_using_Biotsavart_28_june(X,Y,Z,20,ch29_coil_array_sub_FOV1.tra_coil(i,j).trace);
        data_coil_trace{cnt,1} = ch29_coil_array_sub_FOV1.tra_coil(i,j).trace;
        cnt=cnt+1;
    end
end

% tStart = cputime;
% Bz=cal_Bz_Biotsavart_HHT(X',Y',Z',ones(29),data_coil_trace);
% execute_time = cputime - tStart

f = @(c) norm(sum(cal_Bz_Biotsavart_HHT(X',Y',Z',c, data_coil_trace),2)-B0_no_shim',1);
c0 = rand(29)*20;

% options = optimset('Display','iter','PlotFcns',@optimplotfval );
% options = optimoptions('fminunc','Display','iter','PlotFcn', 'optimplotx' ,'Algorithm','quasi-newton');
% [x,fval,exitflag,output] = fminsearch(f,c0,options);
% [x,fval,exitflag,output] = fminunc(f,c0,options)

lb = -60*ones(29,1);
ub = 60*ones(29,1);
options = optimoptions('simulannealbnd','PlotFcns', {@saplotbestx,@saplotbestf,@saplotx,@saplotf},'Display','iter' ...
    ,'ReannealInterval',500);
[c,fval,exitflag,output] = simulannealbnd(f,c0,lb,ub,options)

% options = optimoptions('particleswarm','SwarmSize',29,'HybridFcn',@fmincon,'UseParallel',true,'Display','iter','PlotFcns','pswplotbestf');
% [c,fval,exitflag,output,points] = particleswarm(f,29,lb,ub,options)