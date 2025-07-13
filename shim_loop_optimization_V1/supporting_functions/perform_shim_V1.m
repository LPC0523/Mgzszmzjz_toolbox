
% function [shimmed,amps,std_unshimmed,std_shimmed] = perform_shim_V1(unshimmed,mask,coil,max_current,total_current)
%
% Basic constrained optimization wrapper code for B0 shimming calculation
%
%
% Jason Stockmann - June 2014
% jaystock@nmr.mgh.harvard.edu
%
% Version 1.0
%
% Information: 
%
%  'unshimmed' are (nx, ny, nz) input B0 field maps in Hz
%
%  'coil' is a (nx, my, nz, nq) matrix containing the B0 field maps 
%  of the shim coils 
%
%  'max_current' is the maximum allowable current
%
%  'amps' is the optimal current in each coil element
%
%  'shimmed' is the resulting (nx, ny, nz) field map after optimal shim 
% currents are applied
%
%   The code also returns the standard deviation of the unshimmed and
%   shimmed B0 maps
%

function [shimmed,amps,std_unshimmed,std_shimmed] = perform_shim_V1(unshimmed,mask,coil,max_current,total_current)

unshimmed(isnan(unshimmed)) = 0;

nx = numel(coil(:,1,1,1));
ny = numel(coil(1,:,1,1));
nz = numel(coil(1,1,:,1));
nq = numel(coil(1,1,1,:));

unshimmed_vec = mask2vec(unshimmed,mask);
A = mask2vec(coil,mask);
A_all = reshape(coil,[nx*ny*nz nq]);
 

% set tolerances to ensure convergence 
options = optimset('DerivativeCheck','off','GradObj','on','Display','iter-detailed','MaxFunEvals',7000,'TolX',2e-8,'TolCon',1e-7);

% perform shim

 amps = fmincon( @(amps)shim_fun_V1(amps,A,unshimmed_vec),zeros(nq,1),[],[],[],[],-max_current*ones(nq,1),...
     max_current*ones(nq,1),@(amps)first_order_norm(amps,total_current),options);

% amps = fmincon( @(amps)shim_fun(amps,A,unshimmed_vec),zeros(nq,1),[],[],[],[],-max_current*ones(nq,1), max_current*ones(nq,1),[],options);


% amps_pinv = A\unshimmed_vec

shimmed = A_all*amps;

shimmed = reshape(shimmed,[nx ny nz]);

shimmed = (unshimmed - shimmed).*double(mask);


shimmed(isnan(shimmed)) = 0;


std_unshimmed = calc_std(unshimmed,mask);
std_shimmed = calc_std(shimmed,mask);

% flip polarity of current setting for shim_fun (double check on shim_fun_std!!!)
amps = double(-amps);

disp(['standard deviation before shim is ',num2str(mean(std_unshimmed)), ' and after shim is ',num2str(mean(std_shimmed))])








