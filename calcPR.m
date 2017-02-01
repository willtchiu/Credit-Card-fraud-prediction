function [prec, recall] = calcPR(p, y)

% Calculate Precision/Recall
true_pos = sum(p & y);
false_pos = sum((p - y) == 1);
false_neg = sum((p - y) == -1);

prec = true_pos/(true_pos+false_pos);
recall = true_pos/(true_pos+false_neg);

end