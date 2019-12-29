function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

C_list = [0.01 0.03 0.1 0.3 1 3 10 30];
sigma_list = [0.01 0.03 0.1 0.3 1 3 10 30];

cost = 0; %cost_previous = 0;

optimization_diagnostics = zeros((size(C_list,2))^2,3); index = 1;

for i = 1:size(C_list,2)
    for j = 1:size(sigma_list,2)
        C_train = C_list(i); 
        sigma_train = sigma_list(j);
        
        Model= svmTrain(X, y, C_train, @(x1, x2) gaussianKernel(x1, x2, sigma_train));
        
        prediction  = svmPredict(Model,Xval);
        cost = mean(double(prediction ~= yval));
        
        %Fill optimization_diagnositics
        optimization_diagnostics(index,1) = C_train; optimization_diagnostics(index,2) = sigma_train; optimization_diagnostics(index,3) = cost;
        index = index + 1;
        
        %cost_previous = cost;
    end
end

[a,b] = min(optimization_diagnostics(:,3));
C = optimization_diagnostics(b,1); sigma = optimization_diagnostics(b,2);
            
            

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%







% =========================================================================

end
