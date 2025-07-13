%MRIR_PHASE_UNWRAP__PRELUDE
% function phz_unwrap = mrir_phase_unwrap__prelude(img, varargin)
%
% 
% phz_unwrap = mrir_phase_unwrap__prelude(phz)
%
%  input should be in the form:
%  img = mag.*exp(1i*fm*shim.delta_TE*2*pi);
%
% this code began as a part of the function "mrir_temperature_map.m".

% jonathan polimeni <jonp@nmr.mgh.harvard.edu>, 2008/may/21
% modified by jason stockmann <jaystock@nmr.mgh.harvard.edu>, 2015/sept/01
% $Id$
%**************************************************************************%
function phz_unwrap = mrir_phase_unwrap__prelude(img, varargin)

  VERSION = '$Revision: 1.6 $';
  if ( nargin == 0 ), help(mfilename); return; end;


  % TODO: use global variable PRELUDE as a matlab environment-like variable
  prelude = '/usr/local/fsl/bin/prelude';


  %==--------------------------------------------------------------------==%
  % phase unwrapping through calls to FSL's "prelude"

  tmpdir = mrir_sysutil__tempdir(mfilename);

  
  
  save_nii(make_nii(squeeze(  abs(img))), sprintf('%s/abs.nii', tmpdir));
  save_nii(make_nii(squeeze(angle(img))), sprintf('%s/phz.nii', tmpdir));

  if length(varargin) == 1
     mask = varargin{1};
      save_nii(make_nii(squeeze(mask)), sprintf('%s/mask.nii', tmpdir));
      disp('saving mask...')

  
      [status, result] = system(sprintf(...
           '%s -v -a %s/abs.nii -p %s/phz.nii  -u %s/phz_unwrap.nii -m %s/mask.nii', ...
            prelude, tmpdir,       tmpdir,          tmpdir,     tmpdir));
        
  else
      disp('test test test')
         [status, result] = system(sprintf(...
           '%s -v -a %s/abs.nii -p %s/phz.nii  -u %s/phz_unwrap.nii', ...
            prelude, tmpdir,       tmpdir,          tmpdir));
  

  end
  
% 
%   [status, result] = system(sprintf(...
%       '%s -v -a %s/abs.nii -p %s/phz.nii -s -u %s/phz_unwrap.nii', ...
%        prelude, tmpdir,       tmpdir,          tmpdir));

  if ( status ), error(result); end;

  disp(sprintf('==> [%s]:  prelude completed phase unwrapping; reading data...', mfilename));  
  
  
  [status, result] = system(sprintf(...
      'gunzip -c %s/phz_unwrap.nii.gz > %s/phz_unwrap.nii', ...
                 tmpdir,                tmpdir));

  if ( status ), error(result); end;

  phz_unwrap = getfield(load_nii(sprintf('%s/phz_unwrap.nii', tmpdir)), 'img');


  %% TODO: read in mask used by "prelude" (crashes for some reason)


  return;


  %************************************************************************%
  %%% $Source: /home/jonp/cvsroot/dotfiles/emacs,v $
  %%% Local Variables:
  %%% mode: Matlab
  %%% fill-column: 76
  %%% comment-column: 0
  %%% End:
