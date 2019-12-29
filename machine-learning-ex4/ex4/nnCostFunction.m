function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
%%  Forward propagation to obtain h_theta_x

% Adding bias units to each layer
X = [ones(m,1) X];

%Calculating output at each layer
layer_1_output = sigmoid(Theta1*(X.'));
layer_1_output = layer_1_output.';
%n = size(layer_1_output, 1);
layer_1_output = [ones(m,1) layer_1_output];
layer_2_output = sigmoid(Theta2*(layer_1_output.'));
layer_2_output = layer_2_output.';

%Vector 'p' will store NN hypothesis values

p = zeros(size(X, 1), 1);

% for a = 1:num_labels
%     [value,index] = max(layer_2_output(a,:));
%     p(a) = index;
% end

% for i = 1:m
%     for j = 1:num_labels
%         


 for i  = 1:num_labels
    y_t = (y == i);    %y transformed to zeros and ones
    J = J + sum(((1/m) *((-y_t).*log(layer_2_output(:,i)) - (1 - y_t).*(log(1 - layer_2_output(:,i))))));
 end
 
 % Adding regularization terms to cost function
 
 for i = 2:size(Theta1,2)
     for j = 1:size(Theta1,1)
 J = J + (lambda/(2*m))*(Theta1(j,i))^2;
     end
 end
 
for i = 2:size(Theta2,2)
     for j = 1:size(Theta2,1)
 J = J + (lambda/(2*m))*(Theta2(j,i))^2;
     end
 end
 
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%

 
for i = 1 : m
    
 %  Step 1 - Forward propagation
    a_1 = X(i,:);
    z_2 = (a_1*Theta1.');
    a_2 = sigmoid(z_2);
    % Adding bias terms to a_2( in a_1 already added by manipulation of vector X )
    a_2 = [1 a_2]; 
    a_3 = sigmoid(a_2*Theta2.');
    
  % Step 2 - Calculate del_3
    y_set = zeros(1,num_labels); 
    if y(i) ~= 10
        y_set(y(i)) = 1;
    else
        y_set(10) = 1;
    end
    
    del_3  = a_3 - y_set;
    
 % Step 3 - Calculate del_2    
 

 del_2 = (Theta2.'*del_3.'); %del_3 assumed to be column vector in pdf so transposed here  
 z_2 = [ones z_2]; % Bias unit added to z_2
 del_2 = (Theta2.'*del_3').* sigmoidGradient(z_2).';

 % Step 4 - Calculate Del_l(accumulator)
   % Making deltas into column vectors
   % del_3 = del_3';
del_2 = del_2(2:end)   
   if i == 1
 Del_1 = del_2*(a_1);
 Del_2 = del_3.'*a_2;  % Again transpositions used for index agreement
   else
 Del_1 = Del_1 + del_2*(a_1);
 Del_2 = Del_2 + del_3.'*(a_2);  % Again transpositions used for index agreement
   end
end

% Step 5 - obtain unregluarized gradient for NN cost function and put them in Theta1_grad and Theta2_grad

Theta1_grad = (1/m)*Del_1;
Theta2_grad = (1/m)*Del_2;

% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%



















% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
