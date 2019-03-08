function fea_extract_train(maindir)

%% file path and paras
norp = 64;
wtmaindir = 'fea\train';

typestr = {'fake_part', 'real_part'};

if ~exist(wtmaindir, 'dir')
    mkdir(wtmaindir);
end

%% feature extraction
for typeidx = 1 : size(typestr,2)   % for real or attack
    
    fprintf('**********************************************************************************\n');
    fprintf('**********************   %s   **********************\n', typestr{typeidx});
    fprintf('**********************************************************************************\n\n\n');
    
    sdir1 =  dir( fullfile( maindir, typestr{typeidx} ) );
    for sdiridx1 = 1 : length(sdir1)    % for each identity in real or attack part
        
        if( isequal( sdir1(sdiridx1).name, '.' ) || ...
                isequal( sdir1(sdiridx1).name, '..' ) || ...
                ~sdir1(sdiridx1).isdir )
            continue;
        end
        
        sdir2 = dir( fullfile( sdir1(sdiridx1).folder, sdir1(sdiridx1).name ) );
        for sdiridx2 = 1 : length(sdir2)    % for each sub type of an identity
            
            if( isequal( sdir2(sdiridx2).name, '.' ) || ...
                    isequal( sdir2(sdiridx2).name, '..' ) || ...
                    ~sdir2(sdiridx2).isdir )
                continue;
            end
            
            wtsubdir = fullfile( wtmaindir, typestr{typeidx}, sdir1(sdiridx1).name, ...
                sdir2(sdiridx2).name );
            if ~exist(wtsubdir, 'dir')
                mkdir(wtsubdir);
            end
            
            sdir3 = dir( fullfile( sdir2(sdiridx2).folder, sdir2(sdiridx2).name ) );
            for sdiridx3 = 1 : length(sdir3)    % for each camera
                
                if( isequal( sdir3(sdiridx3).name, '.' ) || ...
                        isequal( sdir3(sdiridx3).name, '..' ) || ...
                        ~sdir3(sdiridx3).isdir )
                    continue;
                end
                
                imgpath = fullfile( sdir3(sdiridx3).folder, sdir3(sdiridx3).name, '*.jpg' );
                imgfile = dir(imgpath);
                fprintf( '**************   train set / %s / %s / %s / %s --- %d images   **************\n', ...
                    typestr{typeidx}, sdir1(sdiridx1).name, sdir2(sdiridx2).name, sdir3(sdiridx3).name, length(imgfile) );
                fea = [];
                for n = 1:length(imgfile)
                    
                    fprintf( 'n = %d, %s\n', n, imgfile(n).name);
                    
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
                    tpfea = [];
                    
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
                    tpfea = [tpfea, blkgslbpfea];
                    blkgslbpfea = lbp_blk(NorImage1(:,:,2), 1, 8, 'u2', 'h', slw, slw);
                    tpfea = [tpfea, blkgslbpfea];
                    blkgslbpfea = lbp_blk(NorImage1(:,:,3), 1, 8, 'u2', 'h', slw, slw);
                    tpfea = [tpfea, blkgslbpfea];
                    blkgslbpfea = lbp_blk(NorImage2(:,:,1), 1, 8, 'u2', 'h', slw, slw);
                    tpfea = [tpfea, blkgslbpfea];
                    blkgslbpfea = lbp_blk(NorImage2(:,:,2), 1, 8, 'u2', 'h', slw, slw);
                    tpfea = [tpfea, blkgslbpfea];
                    blkgslbpfea = lbp_blk(NorImage2(:,:,3), 1, 8, 'u2', 'h', slw, slw);
                    tpfea = [tpfea, blkgslbpfea];
                    
                    % ccolbp
                    F = ColorCoALBP(NorImage1, 1, 2, 1, 'h');
                    for fi = 1 : size(F, 2)
                        tpfea = [tpfea, (F(:, fi))'];
                    end
                    F = ColorCoALBP(NorImage1, 2, 4, 1, 'h');
                    for fi = 1 : size(F, 2)
                        tpfea = [tpfea, (F(:, fi))'];
                    end
                    F = ColorCoALBP(NorImage2, 1, 2, 1, 'h');
                    for fi = 1 : size(F, 2)
                        tpfea = [tpfea, (F(:, fi))'];
                    end
                    F = ColorCoALBP(NorImage2, 2, 4, 1, 'h');
                    for fi = 1 : size(F, 2)
                        tpfea = [tpfea, (F(:, fi))'];
                    end
                    
                    fea = [fea; tpfea];
                    
                end
                
                % save feature vector
                wtfeafilename = strcat( sdir3(sdiridx3).name, '.mat' );
                wtfeapathname = fullfile( wtsubdir, wtfeafilename );
                save(wtfeapathname, 'fea', '-v7.3');
                
            end
            
        end
        
    end
    
end

end
