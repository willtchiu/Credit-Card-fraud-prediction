clear; close all; clc

% Load credit card fraud data
data = csvread('creditcard.csv');

% Trim out column headers
data = data(2:end, :);

[m_ent, n_ent] = size(data);

X = data(:, 1:n_ent-1);
Y = data(:, n_ent);

% Normalize all features
X_mean = mean(X);
X_norm = (X.-X_mean)./std(X);
% X_norm = (X.-X_mean);

% Only feature scale 'amount'
%X_norm(:, n_ent-1) = X_norm(:, n_ent-1)./std(X(:, n_ent-1));
X_norm = [ones(m_ent, 1) X_norm];

data_norm = [X_norm, Y];

under_sample_data = underSample(data_norm, Y);

[m_under, n_under] = size(under_sample_data);

% Check our ratios
percent_fraud = size(find(under_sample_data(:, n_under) == 0), 1)/size(under_sample_data, 1);
percent_good = size(find(under_sample_data(:, n_under)), 1)/size(under_sample_data, 1);
fprintf(['Percentage of fraudulent transactions: %f \n'], percent_fraud);
fprintf(['Percentage of good transactions: %f \n'], percent_good);
fprintf(['Total number of transactions in under_sample data: %d \n'], size(under_sample_data, 1));


% Drop time feature? under_sample_data(1)
% under_sample_data = under_sample_data(:, 2:n);

% Split data into train, CV, and test
Xtrain = under_sample_data(1:round(m_under*0.6), 1:n_under-1);
Ytrain = under_sample_data(1:round(m_under*0.6), n_under);

Xval = under_sample_data(round(m_under*0.6)+1:round(m_under*0.8), 1:n_under-1);
Yval = under_sample_data(round(m_under*0.6)+1:round(m_under*0.8), n_under);

Xtest = under_sample_data(round(m_under*0.8)+1:end, 1:n_under-1);
Ytest = under_sample_data(round(m_under*0.8)+1:end, n_under);

initial_theta = zeros(n_ent, 1);

lambda = 3000;
% Find optimal parameters of theta with trainLambda
%trainLambda(Xtrain, Xval, Ytrain, Yval, initial_theta)

% Compute on test set data
options = optimset('GradObj', 'on', 'MaxIter', 500);
[theta, cost] = ...
    fminunc(@(t)(costFunctionReg(t, Xtrain, Ytrain, lambda)), initial_theta, options);

p = predict(theta, Xtest);
[prec_under, recall_under] = calcPR(p, Ytest);

fprintf('Precision (under_sample; test): %f\n', prec_under);
fprintf('Recall (under_sample; test): %f\n\n', recall_under);

% Now fit our model on the whole data set
%trainLambda(Xtrain, X_norm, Ytrain, Y, initial_theta)
p = predict(theta, X_norm);

[prec, recall] = calcPR(p, Y);

% Calculate standard accuracy
acc = size(find(p == Y), 1)/size(Y, 1);

fprintf('Precision for entire data set: %f\n', prec);
fprintf('Recall for entire data set: %f\n', recall);
fprintf('Accuracy for entire data set: %f\n', acc);
