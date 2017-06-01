function [trainm, trainl, testm, testl] = dataloader()
    directory = '../data/';
    trainm = importdata(strcat(directory,'x_train', '.mat'));
    trainl = importdata(strcat(directory,'y_train', '.mat'));
    testm = importdata(strcat(directory,'x_test', '.mat'));
    testl = importdata(strcat(directory,'y_test', '.mat'));
end