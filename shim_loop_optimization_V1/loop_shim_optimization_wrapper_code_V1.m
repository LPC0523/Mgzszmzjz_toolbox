% This code simulates B0 shim performance for a subarray of shim coils over
% a set of in vivo brain B0 field maps.  
%
% The code loads the field maps, performs masking and phase unwrapping, 
% and then simulates shim performance of the coil array based using the 
% standard deviation of delta_B0 over the volume (sigma_B0_global).
% The code uses a genetic algorithm to select an optimal
% subset of coils, or subarray, from a "fully-populated" coil geometry.
% For example, the code may begin with a 39 channel array tiled over head
% covering the brain, and then select the subarray of 8 coils that produces
% the minimum sigma_B0_global, defined as the standard deviation of
% delta_B0 over all slices in the shimmed B0 field maps.
%
% The code requires access to "fmincon" in the Optimization Toolbox and
% "ga" in the Global Optimization Toolbox
%
% Please see 'READ_ME_multi_coil_shim_simulations.txt' for more details
%
% Jason Stockmann, MGH
% 
% May 2016
% Version 1.0
%
% For bug reports, comments, and suggestions, please contact:
% jaystock@nmr.mgh.harvard.edu
% 
% Disclaimer:  The author and Massachusetts General Hospital are not liable
% for any damages that occur in connection with using, modifying, or 
% distributing this software.  The software is not intended for diagnostic
% purposes.

clear all

% change to current directory
curr_path = fileparts(mfilename('fullpath'))
cd(curr_path)

str_OS = computer;  % returns MACI64 for mac, GLNXA64 for Linux, or PCWIN64 for Windows
sep = filesep;   % determine if / or \ character is used for file paths

% =========================================================================
%% INPUT PARAMETERS
% Options for loading field maps:  
% set to '1' to load field map datasets from DICOM files;  set to '0' to
% load from data structure field_maps.mat
% the data structure must include 'mag' for magnitude, 'unshimmed' for the
% baseline B0 field maps in Hz, and 'mask' for the brain mask.  Size for each
% matrix should be [nx, ny, nz, number_of_subjects]
%
% Option 1 will automatically try to use FSL to perform masking and phase
% unwrapping on the field maps
load_from_dicoms = 0;   
% identifier numbers assigned to subjects.  A few representative 3T field 
% maps are included with this software.  Dimension must be equal to
% number_of_subjects
shim.subj = [4 5 6 7 9 10];   

% field map processing parameters (if loading data form DICOM files)
shim.delta_TE = .00246;     % delta TE for field mapping sequence
shim.bet_f = .36;           % threshold for brain extraction tool mask.  Tweak if desied.

% set allowable shim currents
shim.max_current = 2.5;   % max current per loop (Amps)
shim.total_current = 35;  % max total current used in entire subarray (Amps)

%%%%%%%%% SPECIFY FULLY-POPULATED COIL ARRAY AND SUBARRAY SIZE %%%%%%%%%%%%
         coil_field_map_file = 'hybrid_rf_shim_coil_faceloops_39ch.mat'
    % other coil array options included with software package:
    %
    %     coil_field_map_file = 'b1_8ch_save'
    %     coil_field_map_file = 'b1_32ch_save'
    %     coil_field_map_file = 'b1_48ch_save'
    %     coil_field_map_file = 'b1_64ch_save'
    %     coil_field_map_file = 'b1_128ch_save'
    %     coil_field_map_file = 'b1_48ch_cylindrical_save'  % has somewhat better global shimming performance than helmet-style coils

 num_channels_constrained = 20;   % number of channels selected for optimal subarray
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%% PLOTTING PARAMETERS
plot_switch = 1;    % set to 1 to enable plotting, set to 0 to disable plotting
representative_slices = [3 8 13 18 23 28 33 38 43]; % for plotting purposes
plot_range = [-100 100];
mag_range = [100 600];
    hist_range = [-40 40];   num_bins = 75;   hist_max = 10e3;  plot_font = 24;

% =========================================================================
%% LOAD COIL ARRAY B0 CALCULATED FIELD MAPS 
    load(coil_field_map_file);
    shim.output_folder = ['.',sep,'outputs',sep,'global_shim_',num2str(numel(b1_z(1,1,1,:))),'ch_array'];
    mkdir(shim.output_folder);

    shim.coil = 42.57e6*b1_z;  clear b1_z b1_x b1_y
    shim.coil(isnan(shim.coil)) = 0;
    shim.coil = single(shim.coil);                   
    
    nc = numel(shim.coil(1,1,1,:));
% =========================================================================

% =========================================================================
%% LOAD AND PROCESS FIELD MAPS %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% An arbitrary number of field maps can be loaded here.   The shim.subj
% variable holds an integer that describes the subject number and is used
% for refering to the subjects as part of a larger data series.

if load_from_dicoms == 1

        shim.multi = 1;
        mag_folder = cell(6,1); 
        fm_folder = cell(6,1);


        for ss=1:numel(shim.subj)
          mag_folder{ss} = (['.',sep,'supporting_data',sep,'subject_',num2str(shim.subj(ss)),sep,'GRE_FIELD_MAP_BASELINE_MAG']);  
          fm_folder{ss} = (['.',sep,'supporting_data',sep,'subject_',num2str(shim.subj(ss)),sep,'GRE_FIELD_MAP_BASELINE_PHASE']);
        end


        shim.num_subj = numel(mag_folder);


        %% PROCESS FIELD MAPS 
        for ss = 1:numel(mag_folder)
                shim.mag(:,:,:,ss) = dicom_fm_import(mag_folder{ss});  
                shim.phase(:,:,:,ss) = dicom_fm_import(fm_folder{ss},shim.delta_TE,1);

                % BRAIN EXTRACTION TOOL FOR MASKING
                shim.mask(:,:,:,ss) = mrir_brain_mask__BET(shim.mag(:,:,:,ss), shim.bet_f);
                shim.mask(:,:,:,ss) = imerode(shim.mask(:,:,:,ss),[0 1 0; 1 1 1; 0 1 0]);
                shim.mask(:,:,:,ss) = imerode(shim.mask(:,:,:,ss),[0 1 0; 1 1 1; 0 1 0]);

                % PHASE UNWRAPPING WITH PRELUDE
                shim.unwrapped(:,:,:,ss) = mrir_phase_unwrap__prelude(shim.mag(:,:,:,ss).*exp(1i*shim.phase(:,:,:,ss)*shim.delta_TE*2*pi), shim.mask(:,:,:,ss));
                shim.unshimmed(:,:,:,ss) = single(shim.unwrapped(:,:,:,ss)./(shim.delta_TE*2*pi));
        end
else
    load field_maps.mat
    shim.unshimmed = unshimmed;  shim.mag = mag; shim.mask = mask;  clear mag mask unshimmed
    shim.num_subj = numel(shim.mag(1,1,1,:));
end

%% ========================================================================


    nx = numel(shim.mag(:,1,1,1));  ny = numel(shim.mag(1,:,1,1));  nz = numel(shim.mag(1,1,:,1));
 

if plot_switch == 1
% present sagittal and coronal views for one representative subject
subject_to_plot = 2;
    temp1 = squeeze(double(shim.mask(:,end/2,:,subject_to_plot)));  temp1(temp1 == 0) = nan;
    temp2 = squeeze(double(shim.mask(end/2,:,:,subject_to_plot)));  temp2(temp2 == 0) = nan;

   figure(50), 
   subplot(2,2,1),imagesc(rot90(squeeze(shim.mag(:,end/2,:,subject_to_plot))),[0 .5*max(max(max(shim.mag(:,:,:,subject_to_plot))))]),axis off,axis image,set(gca,'FontSize',16),title('magnitude sagittal midline')
    subplot(2,2,2),imagesc(rot90(squeeze(shim.mag(end/2,:,:,subject_to_plot))),[0 .5*max(max(max(shim.mag(:,:,:,subject_to_plot))))]),axis off,axis image,set(gca,'FontSize',16),title('magnitude coronal')
    % imagesc2 sets nan values to white, making it easier to plot masked-out regions as white instead of green in the 'jet' colormap
    subplot(2,2,3),imagesc2(rot90(squeeze(shim.unshimmed(:,end/2,:,subject_to_plot)).*temp1),[-100 100]),colorbar,axis off,colormap(jet),set(gca,'FontSize',16),axis image,hcb=colorbar,title(hcb,'Hz'),title('sagittal midline \Delta B_0 map')
    subplot(2,2,4),imagesc2(rot90(squeeze(shim.unshimmed(end/2,:,:,subject_to_plot)).*temp2),[-100 100]),colorbar,axis off,colormap(jet),set(gca,'FontSize',16),axis image,hcb=colorbar,title(hcb,'Hz'),title('coronal \Delta B_0 map')   
end




% =========================================================================
%% PERFORM SUBARRAY OPTIMIZATION
% SYNTAX:    X = ga(FITNESSFCN,NVARS,A,b,[],[],lb,ub,NONLCON,INTCON,options)
opts = gaoptimset('Vectorized','off','PlotFcns',{@gaplotbestf,@gaplotstopping});

coils_used = zeros(1,nc);  % initialize vector that stores the active subarray elements

    
    
  FitnessFunction = @(coils_used)shim_optimization_objective_function_ga_V1(coils_used,shim,shim.subj,shim.num_subj);
  ConstraintFunction = @(coils_used)channel_count_constraint_for_ga_V1(coils_used,num_channels_constrained);



% PERFORM GENETIC ALGORITHM OPTIMIZATION
[coils_used,Fval,exitFlag,Output] = ga(FitnessFunction,nc,[],[],[],[],zeros(1,nc),ones(1,nc),ConstraintFunction,[1:1:nc],opts)


fprintf('The number of generations was : %d\n', Output.generations);
fprintf('The number of function evaluations was : %d\n', Output.funccount);
fprintf('The best function value found was : %g\n', Fval);



% =========================================================================
