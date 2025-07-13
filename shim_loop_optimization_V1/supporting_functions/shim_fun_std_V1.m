
function [f df]= shim_fun_std(x,A,field_map)      
%function [f df]= shim_fun(x,A,field_map)      
      
y=A*x-field_map;

n=size(y,1);

f=y'*y - 1/n*sum(y,1)^2;

df=2*A'*y;
df=df - 2/n*sum(y,1).*sum(A,1).';
    
%     y=A*x - field_map;
%     f=y'*y;
%     df=2*A'*y;
    
end
