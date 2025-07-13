% function std_out = calc_std(in, mask)
%
% set 'in' to be the matrix of interest
% use 'mask' as matrix with 1s at points of interest
% 'in' and 'mask' can be 2D or 3D.  if mask is 2D it 
% will be automatically used for all slices.

function std_out = calc_std_whole_brain(in, mask)

mask = logical(mask);

temp = zeros(sum(sum(sum(sum(mask)))),1);

temp2 = in(mask);

temp = temp2(:);


std_out = std(temp);





% % 
% % 
% % if numel(in(1,1,1,:)) == 1
% % 
% %     if numel(squeeze(in(1,1,:))) == 1
% % 
% %         temp = in;
% % 
% %         temp2 = temp(mask);
% %         std_out = std(temp2(:));
% % 
% % 
% % 
% %     else
% %         
% %         ns = numel(squeeze(in(1,1,:)));
% %         for ss=1:ns
% %             temp = in(:,:,ss);
% %             if numel(mask(1,1,:)) == 1
% %                 temp2 = temp(mask);
% %             else
% %                 temp2 = temp(mask(:,:,ss));
% %             end
% %             std_out(ss) = std(temp2(:));
% %         end
% %         
% % 
% %     end
% % 
% % 
% % 
% % else
%     
%     
%    disp('[calc_std]: matrix has more than three dimensions')
%    std_out = 0; 
%    
% end
% 
% 



