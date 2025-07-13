% function [C Ceq] = channel_count_constraint_for_ga_V1(coils_used, num_channels_constrained)
%
% force number of channels used to equal num_channels_constrained with
% tolerance trick --> can't use Ceq with integer constraints in genetic
% algorithm
%
% Jason Stockmann, MGH, May 2016


function [C Ceq] = channel_count_constraint_for_ga_V1(coils_used, num_channels_constrained)

tol = .5;


C(1) = num_channels_constrained - sum(coils_used) - tol;
C(2) = -(num_channels_constrained - sum(coils_used)) - tol;

Ceq = [];