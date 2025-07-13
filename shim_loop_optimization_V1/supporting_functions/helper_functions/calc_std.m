% function std_out = calc_std(in, mask)
%
% set 'in' to be the matrix of interest
% use 'mask' as matrix with 1s at points of interest
% 'in' and 'mask' can be 2D or 3D.  if mask is 2D it 
% will be automatically used for all slices.
%
% 
% slice switch is 1 to calculate std of each slice
% slice switch is 0 to calculate the global std over all slices


function std_out = calc_std(in, mask,slice_switch)

% slice switch is 1 to calculate std of each slice
% slice switch is 0 to calculate the global std over all slices
if nargin == 2
    slice_switch = 1;
end

mask = logical(mask);
mask(isnan(mask)) = 0;
in(isnan(in)) = 0;

if slice_switch == 1
    if numel(in(1,1,1,:)) == 1

        if numel(squeeze(in(1,1,:))) == 1

            temp = in;

            temp2 = temp(mask);
            std_out = std(temp2(:));



        else

            ns = numel(squeeze(in(1,1,:)));
            for ss=1:ns
                temp = in(:,:,ss);
                if numel(mask(1,1,:)) == 1
                    temp2 = temp(mask);
                else
                    temp2 = temp(mask(:,:,ss));
                end
                std_out(ss) = std(temp2(:));
            end


        end



    else


       disp('[calc_std]: matrix has more than three dimensions')
       std_out = 0; 

    end
    
else
    
    temp = in(mask);
    std_out = std(temp(:));
    
end
