function metrics = EvaluationMetrics(predicted_labels, true_labels)
    % Calculate evaluation metrics for multi-class and binary classification tasks
    %
    % Parameters:
    %   predicted_labels - Predicted class vector (N x 1)
    %   true_labels - True class vector (N x 1)
    %
    % Returns:
    %   metrics - Structure containing various evaluation metrics

classes = unique([true_labels; predicted_labels]);  % Unify class representations
num_classes = length(classes);
C = confusionmat(true_labels, predicted_labels, 'Order', classes);


    % % Get number of classes
    % classes = unique(true_labels);
    % num_classes = length(classes);
    % 
    % % Compute confusion matrix
    % C = confusionmat(true_labels, predicted_labels);

    % Initialize variables
    TP = zeros(num_classes, 1);  % True Positives
    FP = zeros(num_classes, 1);  % False Positives
    FN = zeros(num_classes, 1);  % False Negatives
    TN = zeros(num_classes, 1);  % True Negatives

    % Calculate TP, FP, FN, TN
    for i = 1:num_classes
        TP(i) = C(i, i);
        FP(i) = sum(C(:, i)) - TP(i);
        FN(i) = sum(C(i, :)) - TP(i);
        TN(i) = sum(C(:)) - (TP(i) + FP(i) + FN(i));
    end

    % Prevent division by zero errors
    safe_divide = @(a, b) a ./ (b + (b == 0));

    % Calculate Precision, Recall (Sensitivity), F1 Score
    precision = safe_divide(TP, TP + FP);
    recall = safe_divide(TP, TP + FN);  % Recall = Sensitivity
    f1_score = safe_divide(2 * (precision .* recall), precision + recall);

    % Calculate Macro-averaged metrics
    macro_precision = mean(precision);
    macro_recall = mean(recall);
    macro_f1 = mean(f1_score);

    % Calculate Micro-averaged metrics
    micro_precision = safe_divide(sum(TP), sum(TP + FP));
    micro_recall = safe_divide(sum(TP), sum(TP + FN));
    micro_f1 = 2 * (micro_precision * micro_recall) / (micro_precision + micro_recall);

    % Calculate Weighted metrics

    class_counts = sum(C, 2);  % Number of samples per class
    weights = class_counts / sum(class_counts);
    % weights = weights(:);  % Force conversion to column vector
    weighted_precision = sum(weights .* precision);
    weighted_recall = sum(weights .* recall);
    weighted_f1 = sum(weights .* f1_score);

    % Calculate Accuracy
    accuracy = sum(TP) / sum(C(:));

    % Calculate binary classification Specificity, Sensitivity, AUC
    if num_classes == 2
        specificity = safe_divide(TN, TN + FP);  % Specificity
        sensitivity = recall;  % Sensitivity
        fpr = safe_divide(FP, FP + TN);  % FPR
        auc = (sensitivity(1) + specificity(1)) / 2;  % Calculate AUC

        % Record binary classification metrics
        metrics.sensitivity = sensitivity(1);  % Select Sensitivity for first class
        metrics.specificity = specificity(1);
        metrics.fpr = fpr(1);
        metrics.auc = auc;
    else
        metrics.sensitivity = NaN;
        metrics.specificity = NaN;
        metrics.fpr = NaN;
        metrics.auc = NaN;
    end

    % Record all metrics
    metrics.accuracy = accuracy;
    metrics.macro_precision = macro_precision;
    metrics.macro_recall = macro_recall;
    metrics.macro_f1 = macro_f1;
    metrics.micro_precision = micro_precision;
    metrics.micro_recall = micro_recall;
    metrics.micro_f1 = micro_f1;
    metrics.weighted_precision = weighted_precision;
    metrics.weighted_recall = weighted_recall;
    metrics.weighted_f1 = weighted_f1;

    fprintf('Accuracy: %.4f\n', accuracy);
    fprintf('Macro Precision: %.4f\n', macro_precision);
    fprintf('Macro Recall: %.4f\n', macro_recall);
    fprintf('Macro F1 Score: %.4f\n', macro_f1);
    fprintf('Micro Precision: %.4f\n', micro_precision);
    fprintf('Micro Recall: %.4f\n', micro_recall);
    fprintf('Micro F1 Score: %.4f\n', micro_f1);
    fprintf('Weighted Precision: %.4f\n', weighted_precision);
    fprintf('Weighted Recall: %.4f\n', weighted_recall);
    fprintf('Weighted F1 Score: %.4f\n', weighted_f1);

    if num_classes == 2
        fprintf('Sensitivity (Recall): %.4f\n', metrics.sensitivity);
        fprintf('Specificity: %.4f\n', metrics.specificity);
        fprintf('False Positive Rate (FPR): %.4f\n', metrics.fpr);
        fprintf('AUC: %.4f\n', metrics.auc);
    end
end
