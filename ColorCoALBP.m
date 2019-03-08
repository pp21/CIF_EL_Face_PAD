function F = ColorCoALBP(img, s, r, config, mode)

% Reference:
% [1] R. Nosaka, Y. Ohkawa and K. Fukui,
%     "Feature Extraction Based on Co-occurrence
%       of Adjacent Local Binary Patterns", PSIVT 2011.

% Color CoALBP feature extraction of input image,
% the gray-level co-occurrence matrices in original CoALBP
% is replaced by chromatic co-occurrence matrices

% Input parameters:
%   img - image [height x width x depth, 0-255, three channel]
%   s   - scale of LBP radius
%   r   - interval of LBP pair
%   config - confguration on LBP [default:1]
%          1 for plus configuration (+)
%          2 for cross configuration(x)
% 
%  Output parameters:
%   F - Feature vector [ 256 x 4 x channel numbers of img ]

% ColorCoALBP.m is based on CoALBP.m released by Copyright (C) 2011- Ryusuke Nosaka. 
% Please refer to CoALBP.m for further detail

% init
if ~exist('s', 'var')
    s = 1;
end

if ~exist('r', 'var')
    r = 2;
end

if ~exist('config', 'var')
    config = 1;
end

Z = double(img);
[h, w, d] = size(Z);
A = zeros(h-2*s, w-2*s, d); % init intra-channel LBP images

% obtain LBPs at every pixel for each intra-channel
for didx = 1:d
    
    C = Z(1+s:h-s,1+s:w-s,didx);
    X = zeros(4,h-2*s,w-2*s);
    
    if config == 1     % +
        X(1,:,:) = Z(1+s  :h-s  ,1+s+s:w-s+s, didx)-C;
        X(2,:,:) = Z(1+s-s:h-s-s,1+s  :w-s, didx  )-C;
        X(3,:,:) = Z(1+s  :h-s  ,1+s-s:w-s-s, didx)-C;
        X(4,:,:) = Z(1+s+s:h-s+s,1+s  :w-s, didx  )-C;
    elseif config == 2 % x
        X(1,:,:) = Z(1+s-s:h-s-s,1+s-s:w-s-s, didx)-C;
        X(2,:,:) = Z(1+s+s:h-s+s,1+s-s:w-s-s, didx)-C;
        X(3,:,:) = Z(1+s-s:h-s-s,1+s+s:w-s+s, didx)-C;
        X(4,:,:) = Z(1+s+s:h-s+s,1+s+s:w-s+s, didx)-C;
    end
    X=double(X>0);
    A(:, :, didx) = reshape([1,2,4,8]*X(:,:), h-2*s, w-2*s)+1;
    
end

% obtain a intra-channel LBP pair histogram by using chromatic co-occurrence matrices
[hh ww dd] = size(A);
for didx = 1:dd
    
    if didx == dd-1
        nidx = dd;
    else
        nidx = mod(didx+1, dd);
    end
    
    D  = (A(1+r  :hh-r  ,1+r  :ww-r, didx) - 1) * 16;
    Y1 = A(1+r  :hh-r  ,1+r+r:ww-r+r, nidx) + D;  % (0, r)
    Y2 = A(1+r-r:hh-r-r,1+r+r:ww-r+r, nidx) + D;  % (-r, r)
    Y3 = A(1+r-r:hh-r-r,1+r  :ww-r, nidx  ) + D;  % (-r, 0)
    Y4 = A(1+r-r:hh-r-r,1+r-r:ww-r-r, nidx) + D;  % (-r, -r)
    
    F(:,1+(didx-1)*4) = hist(Y1(:), 1:(16*16));
    F(:,2+(didx-1)*4) = hist(Y2(:), 1:(16*16));
    F(:,3+(didx-1)*4) = hist(Y3(:), 1:(16*16));
    F(:,4+(didx-1)*4) = hist(Y4(:), 1:(16*16));

end

if(strcmp(mode,'nh'))
    F = F(:)./sum(F(:));
end

end