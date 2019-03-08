function fea_extract_test(maindir)

%% file path and paras
norp = 64;
wtmaindir = 'fea\test';

if ~exist(wtmaindir, 'dir')
    mkdir(wtmaindir);
end

%% feature extraction
sdir1 =  dir( fullfile(maindir) );
for sdiridx1 = 1 : length(sdir1)    % for each sub dir in test set
    
    if( isequal( sdir1(sdiridx1).name, '.' ) || ...
            isequal( sdir1(sdiridx1).name, '..' ) || ...
            ~sdir1(sdiridx1).isdir )
        continue;
    end
    
    wtsubdir = fullfile( wtmaindir, sdir1(sdiridx1).name );
    if ~exist(wtsubdir, 'dir')
        mkdir(wtsubdir);
    end
    
    imgpath = fullfile( sdir1(sdiridx1).folder, sdir1(sdiridx1).name, '*.jpg' );
    imgfile = dir(imgpath);
    fprintf( '**************   test set / %s --- %d images   **************\n', ...
        sdir1(sdiridx1).name, length(imgfile) );
    for n = 1:length(imgfile)
        
        fprintf( 'test set / %s / n = %d, %s\n', sdir1(sdiridx1).name, n, imgfile(n).name);
        
        % original image
        oimg = imread( fullfile( imgfile(n).folder, imgfile(n).name ) );
        
        % hsv image
        HSVImage = rgb2hsv(oimg);
        HSVImage = im2uint8(HSVImage);
        
        % YCbCr image
        YCbCrImage = rgb2ycbcr(oimg);
        
        % normalization
        NorImage1 = imresize( HSVImage, [norp, norp] );
        NorImage2 = imresize( YCbCrImage, [norp, norp] );
        
        % get feature
        fea = [];
        
        % guided image filtering and gs-lbp_blk
        r = 4;
        eps = 0.1^2;
        gifip = im2double(NorImage1);
        q(:, :, 1) = guidedfilter(gifip(:, :, 1), gifip(:, :, 1), r, eps);
        q(:, :, 2) = guidedfilter(gifip(:, :, 2), gifip(:, :, 2), r, eps);
        q(:, :, 3) = guidedfilter(gifip(:, :, 3), gifip(:, :, 3), r, eps);
        NorImage1 = q;
        gifip = im2double(NorImage2);
        q(:, :, 1) = guidedfilter(gifip(:, :, 1), gifip(:, :, 1), r, eps);
        q(:, :, 2) = guidedfilter(gifip(:, :, 2), gifip(:, :, 2), r, eps);
        q(:, :, 3) = guidedfilter(gifip(:, :, 3), gifip(:, :, 3), r, eps);
        NorImage2 = q;
        slw = 32;
        blkgslbpfea = lbp_blk(NorImage1(:,:,1), 1, 8, 'u2', 'h', slw, slw);
        fea = [fea, blkgslbpfea];
        blkgslbpfea = lbp_blk(NorImage1(:,:,2), 1, 8, 'u2', 'h', slw, slw);
        fea = [fea, blkgslbpfea];
        blkgslbpfea = lbp_blk(NorImage1(:,:,3), 1, 8, 'u2', 'h', slw, slw);
        fea = [fea, blkgslbpfea];
        blkgslbpfea = lbp_blk(NorImage2(:,:,1), 1, 8, 'u2', 'h', slw, slw);
        fea = [fea, blkgslbpfea];
        blkgslbpfea = lbp_blk(NorImage2(:,:,2), 1, 8, 'u2', 'h', slw, slw);
        fea = [fea, blkgslbpfea];
        blkgslbpfea = lbp_blk(NorImage2(:,:,3), 1, 8, 'u2', 'h', slw, slw);
        fea = [fea, blkgslbpfea];
        
        % ccolbp
        F = ColorCoALBP(NorImage1, 1, 2, 1, 'h');
        for fi = 1 : size(F, 2)
            fea = [fea, (F(:, fi))'];
        end
        F = ColorCoALBP(NorImage1, 2, 4, 1, 'h');
        for fi = 1 : size(F, 2)
            fea = [fea, (F(:, fi))'];
        end
        F = ColorCoALBP(NorImage2, 1, 2, 1, 'h');
        for fi = 1 : size(F, 2)
            fea = [fea, (F(:, fi))'];
        end
        F = ColorCoALBP(NorImage2, 2, 4, 1, 'h');
        for fi = 1 : size(F, 2)
            fea = [fea, (F(:, fi))'];
        end
        
        % save feature vector
        [~, fname, ~] = fileparts( imgfile(n).name );
        wtfeafilename = strcat( fname, '.mat' );
        wtfeapathname = fullfile( wtsubdir, wtfeafilename );
        save(wtfeapathname, 'fea', '-v7.3');
        
    end
    
end

end

