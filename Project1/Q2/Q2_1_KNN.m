% load data sets
load('X_test.mat');
load('X_train.mat');
load('y_test.mat');
load('y_train.mat');
y_test = y_test';
y_train = y_train';

% knn parameter
NUM_K = 5;
% train knn model
model_knn = fitcknn(X_train,y_train,'NumNeighbors',NUM_K);
% test sample prediction
predict_knn = predict(model_knn, X_test);
per_knn = sum(predict_knn == y_test) / size(y_test, 1) * 100;
% reporting
fprintf('If, k = %d, KNN Precsion is %.2f%%. \n', NUM_K, per_knn);
