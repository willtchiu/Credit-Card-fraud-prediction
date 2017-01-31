clear; close all; clc

% Load credit card fraud data
data = csvread('creditcard.csv');

% Trim out column headers
data = data(2:end, :);

m = size(data, 1);
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


% Split data into train, CV, and test
%Xtrain = data(2:round(m*0.6), 1:30);
%Ytrain = data(2:round(m*0.6), 31);

%Xval = data(round(m*0.6)+1:round(m*0.8), 1:30);
%Yval = data(round(m*0.6)+1:round(m*0.8), 31);

%Xtest = data(round(m*0.8)+1:end, 1:30);
%Ytest = data(round(m*0.8)+1:end, 31);