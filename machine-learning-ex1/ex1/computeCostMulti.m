function J = computeCostMulti(X, y, theta)
%COMPUTECOSTMULTI Compute cost for linear regression with multiple variables
%   J = COMPUTECOSTMULTI(X, y, theta) computes the cost of using theta as the
%   parameter for linear regression to fit the data points in X and y

% Initialize some useful values
m = length(y); % number of training examples
n = size(X,2); % number of features

H = zeros(m,1); % Hypotheses


    for b = 1:n
        H = H + theta(b)*X(:,b);
    end
    
% You need to return the following variables correctly 
%J_vector = zeros(m,1); % Cost function vector containing cost of each data point 

J_vector = H - y;

%J = sum(J_vector);
J = J_vector;
% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta
%               You should set J to the cost.





% =========================================================================

end
