% load data sets
load('X_test.mat');
load('X_train.mat');
load('y_test.mat');
load('y_train.mat');

% necessary parameter
NUM_SAMPLE = 3500;
NUM_TEST_SAMPLE = 1000;
NUM_NEURON = 25;

size_y_train = size(y_train');
size_y_test = size(y_test');
y_predicts = zeros(NUM_SAMPLE,NUM_NEURON);

% precision of training set
for i = 1:size_y_train
    for j = 1:NUM_NEURON
        if y_train(1, i) == j
            y_predicts(i, j) = 1;
        end
    end
end

% training model using neural network
net = feedforwardnet(NUM_NEURON);
%FOR testing
%net.trainParam.epochs = 1;

net = configure(net, X_train', y_predicts'); view(net)
net = train(net,X_train', y_predicts'); view(net)

res_ANN = net(X_test');
res_ANN = res_ANN';
all_predicts = zeros(NUM_TEST_SAMPLE,1);

% precision of test sample
for i = 1:size_y_test
    max = realmin('double');
    for j = 1:NUM_NEURON
        if max < res_ANN(i, j)
            max = res_ANN(i, j);
            predict = j;
        end
    end
    all_predicts(i,1) = predict;
end

% difference between precision and actual result
diff_predicts = all_predicts - y_test';

% number of matching
num_match = 0;

% count number of matching
for i = 1:1:size_y_test
    if diff_predicts(i,1)==0
        num_match = num_match+1;
    end
end

% reporting
precision = (num_match/NUM_TEST_SAMPLE) * 100;
fprintf('ANN Precision is %.2f%%. \n',precision);