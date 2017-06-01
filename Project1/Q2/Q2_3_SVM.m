% load data sets
load('X_test.mat');
load('X_train.mat');
load('y_test.mat');
load('y_train.mat');
x_test = X_test;
x_train = X_train;
y_test = y_test';
y_train = y_train';

[sample_size, attribute_size] = size(x_test);
[train_rows, train_cols] = size(x_train);

% Support Vector Machine method
% Reference: https://www.mathworks.com/help/stats/fitcsvm.html
kernel = 2;
labels = unique(y_train);
num_labels = numel(labels);

% model_svm = fitcsvm(x_train, y_train, 'KernelFunction', 'polynomial', 'Standardize', true, 'ClassNames', {'1'});
models_svm = cell(num_labels, 1);
y_train_new=zeros(train_rows, num_labels);
for i=1:train_rows
    for j=1:num_labels
        if y_train(i,1) == j
            y_train_new(i, j)=1;
        end;
    end;
end;

for i=1:num_labels
    SVMModel=fitcsvm(x_train, y_train_new(:,i),'KernelFunction', 'polynomial', 'PolynomialOrder',kernel, 'NumPrint', 10000);
    models_svm{i, 1}=SVMModel;
end;

Scores = zeros(sample_size, num_labels);

% create matrix to store prediction of test sample
predict_svm = zeros(sample_size, 1);

for i = 1: sample_size
    for j = 1: num_labels
        % label = num2str(j);
        label_predicate = predict(models_svm{j, 1}, x_test(i,:));
        if(label_predicate == 1)
            Scores(i, j) = Scores(i, j) + 1;
        end
        if(label_predicate == 0)
            Scores(i, j) = Scores(i, j) - 1;
        end
    end
end
% [~,maxScore] = max(Scores,[],2);

for i = 1 :1: sample_size
    max = Scores(i, 1);
    predict_svm(i, 1) = 1;
    for j = 1 :1: num_labels
        if(max < Scores(i, j))
            max = Scores(i, j);
            predict_svm(i, 1) = j;
        end
    end
end

res_svm = predict_svm - y_test;

% number of matching
match_svm = 0;

for i = 1:1:sample_size
    % count number of matching
    if res_svm(i,1) == 0
        match_svm = match_svm + 1;
    end
end

% percentage of matching
per_svm = match_svm / sample_size * 100;
fprintf('SVM Precsion is %.2f%%. \n', per_svm);
