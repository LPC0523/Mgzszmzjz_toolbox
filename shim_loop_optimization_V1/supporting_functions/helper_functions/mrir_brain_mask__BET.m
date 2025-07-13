function mask_bet = mrir_brain_mask__BET(img,f)
%MRIR_MASK_BET
%
% mask_bet = mrir_brain_mask__BET(img,f)

% this code began as a part of the function "mrir_temperature_map.m".

% jonathan polimeni <jonp@nmr.mgh.harvard.edu>, 2008/may/21
%
% modified to work as a wrapper for the FSL brain extraction tool (BET):
% jason stockmann <jaystock@nmr.mgh.harvard.edu>, 2014/feb/11  
% $Id$
%**************************************************************************%

  VERSION = '$Revision: 1.5 $';
  if ( nargin == 0 ), help(mfilename); return; end;


  % TODO: use global variable PRELUDE as a matlab environment-like variable
  bet = '/usr/local/fsl/bin/bet';


  %==--------------------------------------------------------------------==%
  % phase unwrapping through calls to FSL's "prelude"

  tmpdir = mrir_sysutil__tempdir(mfilename);


  save_nii(make_nii(squeeze(  abs(img))), sprintf('%s/abs.nii', tmpdir));

  % set fractional intensity threshold -f lower to increase volume of selected brain
  [status, result] = system(sprintf(...
      ['%s %s/abs.nii %s/out_bet.nii -m -f ',num2str(f),], ...
       bet, tmpdir,       tmpdir));

  if ( status ), error(result); end;

  disp(sprintf('==> [%s]:  BET completed masking; reading data...', mfilename));  
  
  
  [status, result] = system(sprintf(...
      'gunzip -c %s/out_bet_mask.nii.gz > %s/out_bet_mask.nii', ...
                 tmpdir,                tmpdir));

  if ( status ), error(result); end;

  mask_bet = getfield(load_nii(sprintf('%s/out_bet_mask.nii', tmpdir)), 'img');


  %% TODO: read in mask used by "prelude" (crashes for some reason)


  return;


  %************************************************************************%
  %%% $Source: /home/jonp/cvsroot/dotfiles/emacs,v $
  %%% Local Variables:
  %%% mode: Matlab
  %%% fill-column: 76
  %%% comment-column: 0
  %%% End:
