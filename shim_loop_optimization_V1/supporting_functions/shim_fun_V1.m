% function [f df]= shim_fun(x,A,field_map)      
%
% Penalize least squares error between baseline shim and multi-coil shim
% for choosing optimal shim currents using fmincon
%
% function [f df]= shim_fun(x,A,field_map)      
%
%
% Jason Stockmann and Bastien Guerin, MGH, May 2016
% jaystock@nmr.mgh.harvard.edu
%
% Version 1.0


function [f df]= shim_fun(x,A,field_map)      
      


%% calculate least squares objective function and gradient
     y=A*x - field_map;
     f=(y)'*(y);
      df=2*A'*(y);
    
%       f=(y-mean(y))'*(y-mean(y));
%       df=2*A'*(y-mean(y));
%     
      f = double(f);
      df = double(df);
      
end
