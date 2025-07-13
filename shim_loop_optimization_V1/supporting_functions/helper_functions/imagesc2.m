function h = imagesc2 (varargin)
% a wrapper for imagesc, with some formatting going on for nans
%function h = imagesc2 (x, y, img_data , plot_range)

if length(varargin) == 4
    x = varargin{1};
    y = varargin{2};
    img_data = varargin{3};
    plot_range = varargin{4};
    h = imagesc(x,y,img_data , plot_range);
% plotting data. Removing and scaling axes (this is for image plotting)
elseif length(varargin) == 3
    x = varargin{1};
    y = varargin{2};
    img_data = varargin{3};
    h = imagesc(x,y,img_data);
else
    img_data = varargin{1};
    plot_range = varargin{2};
    h = imagesc(img_data , plot_range);
end


axis image off

% setting alpha values
if ndims( img_data ) == 2
  set(h, 'AlphaData', ~isnan(img_data))
elseif ndims( img_data ) == 3
  set(h, 'AlphaData', ~isnan(img_data(:, :, 1)))
end

if nargout < 1
  clear h
end