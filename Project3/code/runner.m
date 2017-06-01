function Y = runner(kernel, kernel_param)

    % load data
    [training_matrix, training_label ,testing_matrix, testing_label] = dataloader();
    
    % get number of labels
    num_label = size(training_label, 2);
    
    % create a 1*num_label array to store SVMs
    SVMModels = cell(num_label, 1);
    
    for i = 1:num_label
        SVMModels{i, 1} = fitcsvm(training_matrix, training_label(:, i), 'KernelFunction', 'polynomial', 'PolynomialOrder', 2);
    end
end