clc;
clear all;

load('coil_trace.mat');
load('B0_map_valid.mat');

N_rec = 7;
N_tra = 8;

M=zeros(13,200,200,40);

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

fprintf("std(B0_map_5T) = %f\n\n", std(B0_data));
fprintf("norm2(B0_map_5T) = %f\n\n", norm(B0_data,2));
fprintf("norm1(B0_map_5T) = %f\n\n", norm(B0_data,1));
fprintf("max(B0_map_5T) = %f,\t min(B0_map_5T) = %f\n\n", max(B0_data), min(B0_data));

tStart = cputime;
field_basis = cal_Bz_Biotsavart_HHT(X',Y',Z',ones(29,1), data_coil_trace);
execute_time = cputime - tStart
disp(size(field_basis));

% f = @(c) norm(field_basis*c'+B0_no_shim',2);
f = @(c) norm(field_basis*c'+B0_data',1);
% f = @(c) std(field_basis*c'+B0_no_shim');

% f = @(c) std(sum(cal_Bz_Biotsavart_HHT(X',Y',Z',c, data_coil_trace),2)+B0_no_shim',2);
% f = @(c) norm(sum(cal_Bz_Biotsavart_HHT(X',Y',Z',c, data_coil_trace),2)+B0_no_shim',2);
% f = @(c) norm(sum(cal_Bz_Biotsavart_HHT(X',Y',Z',c, data_coil_trace),2)+B0_no_shim',1);
% % c0 = 0.5*(rand(29,1)-0.5);
% c0 = load("particleswarm_results.mat","c").c;
% c0 = c0+rand(1,29)-0.5;


% tStart = cputime;
% % f0 = sum(cal_Bz_Biotsavart_HHT(X',Y',Z',c0, data_coil_trace),2);
% f0 = sum(cal_Bz_Biotsavart_HHT(X',Y',Z',c0, data_coil_trace),2)+B0_no_shim';
% execute_time = cputime - tStart;
% disp(sprintf("execute_time = %f\n", execute_time));
% disp(sprintf("std(f0) = %f\n", std(f0)));
% disp(sprintf("max(f0) = %f,\t min(f0) = %f\n", max(f0), min(f0)));

lb = -5*ones(1,29);
ub = 5*ones(1,29);

% options = optimset('Display','iter','PlotFcns',@optimplotfval );
% [x,fval,exitflag,output] = fminsearch(f,c0,options);


% options = optimoptions('simulannealbnd','PlotFcns', {@saplotbestx,@saplotbestf,@saplotx,@saplotf},'Display','iter','ReannealInterval',200);
% [c,fval,exitflag,output] = simulannealbnd(f,c0,lb,ub,options)

options = optimoptions('particleswarm','SwarmSize',500,'UseParallel',true,'UseVectorized',false,'Display','iter','PlotFcns',@pswplotbestf,'InitialSwarmMatrix',lb + rand(500, 29) .* (ub - lb),'OutputFcn',@pswplotbestf);
[c,fval,exitflag,output,points] = particleswarm(f,29,lb,ub,options)

% c = optimvar("x",29,1);
% fun = fcn2optimexpr(@f,c);
% prob = optimproblem("Objective",fun);
% x.LowerBound = -1;
% x.UpperBound = 1;
% [sol,fval] = solve(prob,"Solver","particleswarm")

% tStart = cputime;
% Bz=cal_Bz_Biotsavart_HHT(X',Y',Z',2*rand(29,1)-1,data_coil_trace);
% execute_time = cputime - tStart
%
% scatter3(X,Y,Z,ones(length(X),1),sum(Bz,2));
% scatter3(X,Y,Z,ones(length(X),1),B0_no_shim);
% alpha 0.05;
% for i = 1:29
%     hold on
%     coil_trace = data_coil_trace{i};
%     plot3(coil_trace(1,:),coil_trace(2,:),coil_trace(3,:));
% end
% grid on

optim = (field_basis*c'+ B0_data')/5;