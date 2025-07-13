
function [C, Ceq] = first_order_norm(amps,total_current)


s = sum(abs(amps));

  s = sum(sqrt(abs(amps).*abs(amps) + 1e-8));


C = s - total_current;



 
Ceq = [];
dCeq = [];