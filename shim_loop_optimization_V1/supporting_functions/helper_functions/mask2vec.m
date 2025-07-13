% function out = mask2vec(in,mask)

function out = mask2vec(in,mask)


if numel(in(1,1,1,:)) == 1
    
temp = in(logical(mask));

out = temp(:);

else  % multi-coil case
   
    out = zeros(sum(sum(sum(mask))),numel(in(1,1,1,:)));
    
    for cc=1:numel(in(1,1,1,:))
       temp = in(:,:,:,cc);
       temp2=temp(logical(mask));
       out(:,cc) = temp2(:);
    end
    
end