clear; close all; clc

% Load credit card fraud data
data = csvread('creditcard.csv');

% Trim out column headers
data = data(2:end, :);

X = data(:, 1:30);
Y = data(:, 31);

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

for i=1:size(rand_good_indices, 1);
    under_sample_data = [under_sample_data; data(rand_good_indices(i), :); data(fraud_indices(i), :)];
end

% Check our ratios
percent_fraud = size(find(under_sample_data(:, 31) == 0), 1)/size(under_sample_data, 1);
percent_good = size(find(under_sample_data(:, 31)), 1)/size(under_sample_data, 1);
fprintf(['Percentage of fraudulent transactions: %f \n'], percent_fraud);
fprintf(['Percentage of good transactions: %f \n'], percent_good);
fprintf(['Total number of transactions in under_sample data: %d \n'], size(under_sample_data, 1));

[m, n] = size(under_sample_data);

% Drop time feature? under_sample_data(1)
% under_sample_data = under_sample_data(:, 2:n);

% Normalize features
under_sample_mean = mean(under_sample_data(:, 1:n-1));
under_sample_y = under_sample_data(:, n);
under_sample_data = (under_sample_data(:, 1:n-1).-under_sample_mean)./std(under_sample_data(:, 1:n-1));
under_sample_data = [under_sample_data under_sample_y];


% Split data into train, CV, and test
Xtrain = under_sample_data(1:round(m*0.6), 1:n-1);
Ytrain = under_sample_data(1:round(m*0.6), n);

Xval = under_sample_data(round(m*0.6)+1:round(m*0.8), 1:n-1);
Yval = under_sample_data(round(m*0.6)+1:round(m*0.8), n);

Xtest = under_sample_data(round(m*0.8)+1:end, 1:n-1);
Ytest = under_sample_data(round(m*0.8)+1:end, n);


% Add intercept term to Xtrain and Xval
[Xtrain_m, Xtrain_n] = size(Xtrain);
[Xval_m, Xval_n] = size(Xval);
Xtrain = [ones(Xtrain_m, 1), Xtrain];
Xval = [ones(Xval_m, 1), Xval];

initial_theta = zeros(Xtrain_n+1, 1);

% Compute initial cost and grad without regularization
[cost, grad] = costFunction(initial_theta, Xtrain, Ytrain);

fprintf('Cost at initial theta (zeros): %f\n', cost);
fprintf('Gradient at initial theta (zeros): \n');
fprintf(' %f \n', grad);

% Find optimal parameters of theta with fminunc