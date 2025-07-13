% Plot results of shim loop optimization
% 
% This script is intended to be run after
% 'loop_shim_optimization_wrapper_code_V1.m' has been run for several
% subjects and coil array sizes.  Results are compared for each specified
% coil sub-array size
%
%
% Questions, comments, and bug reports, please contact:
%
% Jason Stockmann
% Massachusetts General Hospital
% 
% jaystock@nmr.mgh.harvard.edu
%
% Disclaimer:  The author and Massachusetts General Hospital are not liable
% for any damages that occur in connection with using, modifying, or 
% distributing this software.  The software is not intended for diagnostic
% purposes.

 close all, clear all

 
 curr_path = fileparts(mfilename('fullpath'))
 cd(curr_path)

%%%%%%% INPUT SIMULATION DESCRIPTION %%%%%%%%
num_channels_starting = 39;
num_channels_constrained = [8 12 16 20];   % number of channels used for each case simulated
helmet_geometry_text_file = 'array_circle32_plus_faceloops';
subj = [4 5 6 7 9 10];  % number to identify each subject dataset used in the simualtion (and to identify field map output file)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%% PLOTTING OPTIONS %%%%%%%%
plot_helmet_geometry = 1;
% representative_slices = [3 8 13 18 23 28 33 38 43];
slice_to_plot = 26;  
plot_range = [-50 50];

mag_range = [100 600];
hist_range = [-40 40];   num_bins = 75;   hist_max = 10e3;  plot_font = 24;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%  
    

% load(['./output_global/mask.mat']); 
% load(['./output_global/unshimmed.mat']);

%  fm(:,:,:,1) = unshimmed;
%  vox(:,1) = unshimmed(logical(mask));
%  s.ch0 = unshimmed(logical(mask))-mean(unshimmed(logical(mask)));

for qq = 1:numel(subj)
  for nn=1:numel(num_channels_constrained)
    load(['./outputs/global_shim_',num2str(num_channels_starting),'ch_array/shim_opt_output_SUBJECT=',num2str(subj(qq)),'_GLOBAL_62slice_',num2str(num_channels_constrained(nn)),'coils_2.5amps_35total.mat'])
    
    fm(:,:,:,nn+1,qq) = predicted; 
    
%      vox(:,nn+1,qq) = predicted(logical(mask));
    
    temp = unshimmed(logical(mask));
    s.ch0 = temp(:);
    
    temp = predicted(logical(mask));
    eval(['s.ch',num2str(num_channels_constrained(nn)),' = temp(:)']);
    
    if plot_helmet_geometry == 1 & qq == 1
        figure(num_channels_constrained(nn))
        plot_biot_savart_geometry_shared_V1(helmet_geometry_text_file,coils_used),
        view([45, 45]),axis off
    end
    
    pause(.05)
    
    fm_one(:,:,nn,qq) = predicted(:,:,30);
    std_all(nn+1,qq) = std_single_subj;
  end
  std_all(1,qq) = std_before;

end
% fm2 = reshape(fm_one,[100 100*7]);

view([45, 45])

std_overall = mean(std_all,2);

% std_all(end-2) = 45.3;  % fudget factor

%% PLOT RESULTS
 figure(1),plot([0 num_channels_constrained],std_overall,'LineWidth',3),hold on,plot([0 num_channels_constrained],std_all,'bo','LineWidth',2)
 set(gca,'FontSize',28),xlabel('num channels used'), ylabel('std dev. of \DeltaB_0 over 62 slices')


clear std
for vv=1:numel(subj)
     for ss=1:numel(unshimmed(1,1,:,1));
        temp100 = (unshimmed(:,:,ss,vv));
        temp101 = temp100(logical(mask(:,:,ss,vv)));
        temp101(isnan(temp101)) = 0;
        temp102 = temp101(:);
        std_slice(ss,1,vv) = std(temp102-mean(temp102));
     end
    end
for qq=1:numel(num_channels_constrained)
    for vv=1:numel(subj)
        for ss=1:numel(shim.unshimmed(1,1,:,1));

            temp100 = (shim.predicted(:,:,ss,qq,vv));
            temp101 = temp100(logical(shim.mask(:,:,ss,vv)));
            temp101(isnan(temp101)) = 0;
            temp102 = temp101(:);
            std_slice(ss,qq+1,vv) = std(temp102-mean(temp102));
            
        end
    end
end
std_slice(isnan(std_slice)) = 0;
std_slice = squeeze(mean(std_slice,1));


figure(100),
for vv=1:numel(subj)
                 for qq=1:numel(num_channels_constrained)
                     subplot(1,numel(subj),1),imagesc(shim.unshimmed(:,:,slice_to_plot,vv),[plot_range]),colormap(jet),axis image, axis off
                     subplot(1,numel(subj),qq+1),imagesc(shim.predicted(:,:,slice_to_plot,qq,vv)-mean(mean(shim.predicted(:,:,slice_to_plot,qq,vv))),[plot_range]),colormap(jet),axis image, axis off
                 end
                 pause(1)
end

std_slice(1,:) = std_all(1,2);
  

%% plot with both st. devs
figure(2),plot([0 num_channels_constrained],std_all(:,2),'LineWidth',4),hold on,plot([0 num_channels_constrained],std_all(:,2),'bo','LineWidth',4)
set(gca,'FontSize',28,'LineWidth',2.5),xlabel('num channels used'), ylabel('std dev. of \DeltaB_0 over 62 slices')
 hold on,plot([0 num_channels_constrained],mean(std_slice,2),'LineWidth',4),hold on,plot([0 num_channels_constrained],mean(std_slice,2),'ro','LineWidth',4)
ylim([10 30])

  figure(5),plot([0 num_channels_constrained(1:end-1) 40],std_all,'LineWidth',5),hold on,plot([0 num_channels_constrained(1:end-1) 40],std_all,'bo','LineWidth',5)
set(gca,'FontSize',28),xlabel('# channels used'), ylabel('Hz')
title('st. dev. of \DeltaB_0 over 62 slices')

% figure(3), for nn=1:num_channels_constrained+1
%     hold on,nhist(vox(:,nn))
%     h = findobj(gca,'Type','patch');
%     set(h,'FaceColor','r','EdgeColor','w','facealpha',0.75)
% end

figure(4),nhist(s,'linewith',5,'samebins','binfactor',10000,'smooth'),set(gca,'FontSize',30),xlabel('Hz'),ylabel('Prob. Density Func.')
    axis([-150 150 0 .09])

  
  
    