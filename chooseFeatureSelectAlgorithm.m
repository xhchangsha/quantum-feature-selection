function  fea_w = chooseFeatureSelectAlgorithm(X_tr,Y_tr,opt)

currentPath = pwd;        
%% start classification
switch opt.algorithm
    case 'optimize_w_a'
        addpath([currentPath,'\model\intertwine']);
        [~, fea_w] = optimize_w_a(X_tr, Y_tr, 100, 1e-6);
        rmpath([currentPath,'\model\intertwine']);
    
end

