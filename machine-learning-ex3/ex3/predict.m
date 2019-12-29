function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X,1);
n = size(Theta1,1);
num_labels = size(Theta2, 1);

% You need to return the following variables correctly 
X = [ones(m,1) X];
%Layer 1 Calculation
layer_1_output = sigmoid(Theta1*X.');
layer_1_output = layer_1_output';
layer_1_output = [ones(m,1) layer_1_output];

%Layer 2 Calculation
layer_2_output = sigmoid(layer_1_output*Theta2.');

p = zeros(size(X, 1), 1);

for a = 1:m
    [value,index] = max(layer_2_output(a,:));
    p(a) = index;
end

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned neural network. You should set p to a 
%               vector containing labels between 1 to num_labels.
%
% Hint: The max function might come in useful. In particular, the max
%       function can also return the index of the max element, for more
%       information see 'help max'. If your examples are in rows, then, you
%       can use max(A, [], 2) to obtain the max for each row.
%









% =========================================================================


end
