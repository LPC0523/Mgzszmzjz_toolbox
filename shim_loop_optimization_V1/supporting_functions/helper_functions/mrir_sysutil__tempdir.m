function tmpdir = mrir_sysutil_tempdir(basename)
%MRIR_SYSUTIL_TEMPDIR
%
% tmpdir = mrir_sysutil_tempdir(basename)
  
% jonathan polimeni <jonp@nmr.mgh.harvard.edu>, 2008/may/21
% $Id$
%**************************************************************************%

  VERSION = '$Revision: 1.5 $';
  if ( nargin == 0 ), help(mfilename); return; end;


  %==--------------------------------------------------------------------==%
  
  tmpfile = fullfile(tempdir, [basename, '__', num2str(now)], 'CREATED');
  tmpdir = fileparts(tmpfile);

  [status, result] = system(sprintf('mkdir -p %s', tmpdir));
  if ( status ), error(result); end;
  
  [status, result] = system(sprintf('touch %s', tmpfile));
  if ( status ), error(result); end;

  
  return;


  %************************************************************************%
  %%% $Source: /home/jonnyreb/cvsroot/dotfiles/emacs,v $
  %%% Local Variables:
  %%% mode: Matlab
  %%% fill-column: 76
  %%% comment-column: 0
  %%% End:
