clearvars;

opt.algorithm='optimize_w_a';
dataname = 'Data2';
load(fullfile('dataset', [dataname, '.mat']));

X = double(X);
data=[X, Y];     
[~, dim] = size(X);

if strcmp(opt.algorithm , 'optimize_w_a')
    k_dataname = ['k_',dataname]; 
    load(fullfile('dataset', [k_dataname, '.mat']));%Load the data set processed by the kernel circuit
end

all_matrics = cell(1,10);
all_acc = zeros(1, 10);
all_macro_precision = zeros(1, 10);
all_macro_recall = zeros(1, 10);
all_macro_f1 = zeros(1, 10);

no_select_num = 0;

all_indices=crossvalind('Kfold',size(data,1),10);

for k=1:10
    testnum=(all_indices==k);%test set index
    trainnum=~testnum;%train set index
    X_test=X(testnum==1,:);
    X_train=X(trainnum==1,:);
    if strcmp(opt.algorithm , 'optimize_w_a')
        k_X_train=k_X(:,trainnum==1,trainnum==1);
    end
    Y_test=Y(testnum==1,:);
    Y_train=Y(trainnum==1,:);
   
    [~, d] = size(X_test);
    if strcmp(opt.algorithm , 'optimize_w_a')
        fea_w = chooseFeatureSelectAlgorithm(k_X_train,Y_train,opt);
        [T_Weight, T_sorted_features] = sort(fea_w,'descend');
        para.percent = 0.7;
        Num_SelectFeaLY = floor(para.percent*dim);       
        SelectFeaIdx = T_sorted_features(1:Num_SelectFeaLY);
    else 
        fea_w = chooseFeatureSelectAlgorithm(X_train,Y_train,opt);
        SelectFeaIdx = find(fea_w == 1);
    end

    if ~isempty(SelectFeaIdx) 
        X_trainwF = X_train(:,SelectFeaIdx);
        X_testwF = X_test(:,SelectFeaIdx); 
        model = fitcecoc(X_trainwF, Y_train);
        predictedLabels = predict(model, X_testwF);
        metrics = EvaluationMetrics(predictedLabels, Y_test);
        all_matrics{1, k} = metrics;
        all_acc(k) = metrics.accuracy;
        all_macro_precision(k) = metrics.macro_precision;
        all_macro_recall(k) = metrics.macro_recall;
        all_macro_f1(k) = metrics.macro_f1;
    else
        no_select_num = no_select_num+1;
    end 
end

total_acc = sum(all_acc(:))/(10-no_select_num);
total_macro_precision = sum(all_macro_precision(:))/(10-no_select_num);
total_macro_recall = sum(all_macro_recall(:))/(10-no_select_num);
total_macro_f1 = sum(all_macro_f1(:))/(10-no_select_num);

save(['result\',char(dataname),'_svm_',char(opt.algorithm),'_best_result_',num2str(total_acc),'_',num2str(total_macro_precision),'_',num2str(total_macro_recall),'_',num2str(total_macro_f1),'.mat'],'all_matrics', 'all_acc', 'all_macro_precision', 'all_macro_recall', 'all_macro_f1');




