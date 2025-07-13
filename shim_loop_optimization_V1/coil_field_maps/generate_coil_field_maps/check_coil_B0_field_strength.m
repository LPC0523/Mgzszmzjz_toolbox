% calculate B0 field for 1 amp along a line down the center axis of a test loop
% load 32ch coil simulation for comparison
%
% numerically-integrated field from 32ch coil simulation should match
% analyticaly calculation of variable "B" below
%
% Disclaimer:  The author and Massachusetts General Hospital are not liable
% for any damages that occur in connection with using, modifying, or 
% distributing this software.  The software is not intended for diagnostic
% purposes.

diameter = .092;   % diameter in meters

FOV = .240;   % FOV in meters
coil_offset = .12; % coil offset from center
num_points = 100;  % number of points in FOV

x = linspace(-FOV/2,FOV/2,num_points);


amps = 1;   % amps flowing in coil

mu_naught = 4*pi*1e-7;

% field along the center axis line for a circular loop
B = (mu_naught/(4*pi))*2*pi*(diameter/2)^2*amps./((x-coil_offset).^2 + (diameter/2)^2).^1.5;

load b1_32ch_save.mat

figure(10),plot(x,42.57e6*B,'LineWidth',2),xlabel('meters'),hold on,plot(x,42.57e6*b1_y(:,end/2,9,1),'LineWidth',2)
title('comparison of analytical and numerically-integrated field for loop')