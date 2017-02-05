function us_data = underSample(data, Y)

    % Number of minority points (fraud)
    fraud_indices = find(Y);
    num_records_fraud = size(fraud_indices);

    % All good transaction indices
    good_indices = find(Y == 0);

    % Randomly select num_records_fraud from good_indices
    shuffle = randperm(num_records_fraud);
    rand_good_indices = good_indices(shuffle(1:num_records_fraud));
    us_data = [];

    for i=1:size(rand_good_indices, 1)
        us_data = [us_data; data(rand_good_indices(i), :); data(fraud_indices(i), :)];
    end
end