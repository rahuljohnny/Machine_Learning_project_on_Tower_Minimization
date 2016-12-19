function [theta, J_history] = gradientDescentMulti(X, y, theta, alpha, numberOfIterations)
m = length(y); % number of training examples
J_history = zeros(numberOfIterations, 1); %CNU

for iteration = 1:numberOfIterations
    
    hypothesis = X * theta;%ht(x)=g(t^T*x)-of note
    % errors = m*1
    errors = hypothesis .- y;%cost func

    newDecrement = (alpha * (1/m) * errors' * X); %derivitive form
    
    theta = theta - newDecrement';   
    J_history(iteration) = computeCostMulti(X, y, theta);

end

end
