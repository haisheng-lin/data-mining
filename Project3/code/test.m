% parameter setup
kernel = 'polynomial';
% kernel = 'gaussian';
kernel_param = 2;

% load data
[training_matrix, training_label ,testing_matrix, testing_label] = dataloader();

[sample_size, attribute_size] = size(testing_matrix);
label_size = size(training_label, 2);
SVMModels = cell(label_size, 1);

for i = 1:label_size
    SVMModels{i, 1} = fitcsvm(training_matrix, training_label(:, i), 'KernelFunction', kernel, 'PolynomialOrder', kernel_param);
end

% create matrix to store prediction of test sample
predict_svm = zeros(sample_size, label_size);

for i = 1: sample_size
    for j = 1: label_size
        label_predicate = predict(SVMModels{j, 1}, testing_matrix(i, :));
        if(label_predicate >= 0)
            predict_svm(i, j) = 1;
        end
    end
end

andr = and(testing_label, predict_svm);
orr = or(testing_label, predict_svm);

accuracy = sum(andr) / sum(orr);
fprintf('Accuracy is %f\n', accuracy);