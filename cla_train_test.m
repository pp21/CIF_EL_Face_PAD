function cla_train_test()

%% file path and paras
methodstr = {'fea'};
typestr = {'fake_part', 'real_part'};
capstr = {'color', 'depth', 'ir'};
paistr = {'04', '05', '06'};
testlistdir = 'test_public_list.txt';

%% training
for capidx = 1:size(capstr, 2)  % for each camera
    
    for paiidx = 1:size(paistr, 2)    % for each pai
        
        fprintf('capidx = %d, %s, paiidx = %d, %s, training', capidx, capstr{capidx}, paiidx, paistr{paiidx});
        trainfea = [];
        trainlabel = [];
        % for eahc method
        for methodidx = 1 : length(methodstr)
            
            tptrainfea = [];
            for typeidx = 1 : size(typestr,2)   % for real or attack
                
                sdir1 =  dir( fullfile( methodstr{methodidx}, 'train', typestr{typeidx} ) );
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
                                ~sdir2(sdiridx2).isdir || ...
                                ( typeidx == 1 && isempty( strfind( sdir2(sdiridx2).name, paistr{paiidx} ) ) ) )
                            continue;
                        end
                        
                        sdir3 = dir( fullfile( sdir2(sdiridx2).folder, sdir2(sdiridx2).name ) );
                        for sdiridx3 = 1 : length(sdir3)    % for each camera data
                            
                            if ~isempty( strfind( sdir3(sdiridx3).name, capstr{capidx} ) )
                                
                                A = load( fullfile( sdir3(sdiridx3).folder, sdir3(sdiridx3).name ) );
                                A = A.fea;
                                tptrainfea = cat(1, tptrainfea, A);
                                if methodidx == 1
                                    B = zeros( size(A, 1), 1 );
                                    if( typeidx == 2 )
                                        B(:, 1) = 1;
                                    else
                                        B(:, 1) = -1;
                                    end
                                    trainlabel = cat(1, trainlabel, B);
                                end
                                
                            end
                            
                        end
                        
                    end
                    
                    if mod(sdiridx1, 100) == 0
                        fprintf('.');
                    end
                    
                end
                
            end
            trainfea = cat(2, trainfea, tptrainfea);
            clear tptrainfea;
        end
        fprintf(', %d images\n', size(trainlabel, 1));
        
        % model training
        Labels_el = zeros(2, numel(trainlabel));
        Labels_el(1, trainlabel==1) = 1;
        Labels_el(2, trainlabel==-1) = 1;
        Model_el{capidx}{paiidx} = trainSoftmaxLayer(trainfea', Labels_el, 'MaxEpochs', 1000, 'ShowProgressWindow', false);
        clear trainfea; clear trainlabel; clear Labels_el;
        
    end
    
end
save('trained_model.mat', 'Model_el');

%% testing
testlfid = fopen(testlistdir);
testfilelist = textscan(testlfid, '%s %s %s', 'Delimiter', ' ');
fclose(testlfid);
testsfid = fopen( strcat( 'predict_results_test', '.txt'), 'w' );
testidx = 0;
fprintf('Testing');
sdir1 =  dir( fullfile( 'fea', 'test' ) );
for sdiridx1 = 1 : length(sdir1)    % for each sub dir in test set
    
    if( isequal( sdir1(sdiridx1).name, '.' ) || ...
            isequal( sdir1(sdiridx1).name, '..' ) || ...
            ~sdir1(sdiridx1).isdir )
        continue;
    end
    
    feapath = fullfile( sdir1(sdiridx1).folder, sdir1(sdiridx1).name, '*.mat' );
    feafile = dir(feapath);
    for n = 1:3:length(feafile)
        testidx = testidx + 1;
        % for each camera
        testscore = [];
        for capidx = 1:size(capstr, 2)
            A = load( fullfile( feafile(n+capidx-1).folder, feafile(n+capidx-1).name ) );
            testfea = A.fea;
            md1 = Model_el{capidx}{1};
            md2 = Model_el{capidx}{2};
            md3 = Model_el{capidx}{3};
            tptests = md1(testfea')';
            testscore(capidx, 1) = tptests(1);
            tptests = md2(testfea')';
            testscore(capidx, 2) = tptests(1);
            tptests = md3(testfea')';
            testscore(capidx, 3) = tptests(1);
        end
        testfs = min(min(testscore));
        
        fprintf(testsfid, '%s %s %s %.10f\n', testfilelist{1}{testidx}, testfilelist{2}{testidx}, ...
            testfilelist{3}{testidx}, testfs);
        
    end
    fprintf('.');
    
end
fclose(testsfid);
fprintf(', %d images\n', testidx);

end

