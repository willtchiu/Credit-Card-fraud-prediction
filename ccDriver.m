clear; close all; clc

% Load credit card fraud data
data = csvread('creditcard.csv');

% Trim out column headers
data = data(2:end, :);

[m_ent, n_ent] = size(data);

X = data(:, 1:n_ent-1);
Y = data(:, n_ent);

% Normalize features
X_mean = mean(X);
X_norm = (X.-X_mean)./std(X);
X_norm = [ones(m_ent, 1) X_norm];

data_norm = [X_norm, Y];


% Need to under-sample data to create 50/50 ratio of good/fraud

% Number of minority points (fraud)
fraud_indices = find(Y);
num_records_fraud = size(fraud_indices);

% Good class indices
good_indices = find(Y == 0);

% Randomly select num_records_fraud indices from good_indices
shuffle = randperm(num_records_fraud);
rand_good_indices = good_indices(shuffle(1:num_records_fraud));
under_sample_data = [];

for i=1:size(rand_good_indices, 1)
    under_sample_data = [under_sample_data; data_norm(rand_good_indices(i), :); data_norm(fraud_indices(i), :)];
end


[m_under, n_under] = size(under_sample_data);

% Check our ratios
percent_fraud = size(find(under_sample_data(:, n_under) == 0), 1)/size(under_sample_data, 1);
percent_good = size(find(under_sample_data(:, n_under)), 1)/size(under_sample_data, 1);
fprintf(['Percentage of fraudulent transactions: %f \n'], percent_fraud);
fprintf(['Percentage of good transactions: %f \n'], percent_good);
fprintf(['Total number of transactions in under_sample data: %d \n'], size(under_sample_data, 1));


shuffle = randperm(m_under);
under_sample_data = under_sample_data(shuffle, :);

% Drop time feature? under_sample_data(1)
% under_sample_data = under_sample_data(:, 2:n);

% Normalize features
%under_sample_mean = mean(under_sample_data(:, 1:n_under-1));
%under_sample_y = under_sample_data(:, n_under);
%under_sample_data = (under_sample_data(:, 1:n_under-1).-under_sample_mean)./std(under_sample_data(:, 1:n_under-1));
%under_sample_data = [under_sample_data under_sample_y];


% Split data into train, CV, and test
Xtrain = under_sample_data(1:round(m_under*0.6), 1:n_under-1);
Ytrain = under_sample_data(1:round(m_under*0.6), n_under);

Xval = under_sample_data(round(m_under*0.6)+1:round(m_under*0.8), 1:n_under-1);
Yval = under_sample_data(round(m_under*0.6)+1:round(m_under*0.8), n_under);

Xtest = under_sample_data(round(m_under*0.8)+1:end, 1:n_under-1);
Ytest = under_sample_data(round(m_under*0.8)+1:end, n_under);




initial_theta = zeros(n_ent, 1);

lambda = 0.1;
% Find optimal parameters of theta with trainLambda
% trainLambda(Xtrain, Xval, Ytrain, Yval, initial_theta)

% Compute on test set data
options = optimset('GradObj', 'on', 'MaxIter', 500);
[theta, cost] = ...
    fminunc(@(t)(costFunctionReg(t, Xtrain, Ytrain, lambda)), initial_theta, options);

p = predict(theta, Xtest);
[prec_under, recall_under] = calcPR(p, Ytest);

fprintf('Precision (under_sample; test): %f\n', prec_under);
fprintf('Recall (under_sample; test): %f\n\n', recall_under);

% Now fit our model on the whole data set
trainLambda(Xtrain, X_norm, Ytrain, Y, initial_theta)
%p = predict(theta, X_norm);

%[prec, recall] = calcPR(p, Y);

%fprintf('Precision for entire data set: %f\n', prec);
%fprintf('Recall for entire data set: %f\n\n', recall);
