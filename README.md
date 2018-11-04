# CS-433 Project 1 
## Damian Dudzicz, Guillaume Michel, Adrien Vandenbroucque

The structure of the files is the following:

- `proj1_helpers.py` contains the helper functions already given,
 - `implementations.py` contains the required six algorithms implementations,
- `helpers.py` contains functions useful for processing the data and computing losses and gradients of different models,
- `features.py` contains functions useful for crafting new features out of the input data,
- `better_implementations.py` contains the improved version used to obtain the best results,
- `cross_validation.py` contains the k-fold cross validation method and
- `Project1-ML` is a notebook used for determining the best values for the parameters
- `run.py` contains the code to load the data and run one of the models in order to compute predicitons.

The train and test data can be found in the `\data` folder. 

To run each of the implementations change the method called in `run.py`.
By default, the command `python3 run.py` will execute the accelerated_reg_logistic_regression
implementation and compute its accuracy.

> **Note:** The `choose_model()` function in `run.py` let you indicate which model to use. You will also need to provide the parameters of the model.

## More about the files

### `proj1_helpers.py`
This file contains a few given function useful for loading the data, predicting labels, and writing predictions to files. 

### `implementations.py`
This file contains the six functions required for this project:

- `least_squares(y, tx)`
This function uses the given training data and corresponding label to perform a linear regression and return the optimal weights.
It basically computes 
$$w = (X^T X)^{-1}X^Ty$$
where $X$ is the training data and $y$ are the corresponding labels.

- `ridge_regression(y, tx, lambda_)`
This function uses the given training data, corresponding label and regularization term $\lambda$ to perform a ridge regression and return the optimal weights.
It basically computes 
$$w = (X^T X + \lambda I)^{-1}X^Ty$$
where $X$ is the training data, $y$ are the corresponding labels and $\lambda$ is the regularization term.

- `least_squares_GD(y, tx, initial_w, max_iters, gamma)`
This function uses the given training data, corresponding label and parameters needed (number of iterations, step-size and initial weights) to perform the gradient descent algorithm). It then computes the optimal weights by using the gradient descent update rule:
$$w_{k+1} = w_k - \gamma \nabla f(w)$$
where $w_k$ is the vector of weights at iteration $k$, $\gamma$ is the step-size and $f$ is the loss function. Here the loss function is the least squares loss function, i.e. $$f(w) = \frac{1}{2n}\sum_{i=1}^n y_n - x_n^Tw$$, where $n$ is the number of sample in the training data.
With this scheme, the gradient is estimated using all of our training data at each iteration.

- `least_squares_SGD(y, tx, initial_w, max_iters, gamma)`
This function uses the given training data, corresponding label and parameters needed (number of iterations, step-size and initial weights) to perform the gradient descent algorithm. It then computes the optimal weights by using the gradient descent update rule:
$$w_{k+1} = w_k - \gamma \nabla f(w)$$
where $w_k$ is the vector of weights at iteration $k$, $\gamma$ is the step-size and $f$ is the loss function. Here the loss function is the least squares loss function, i.e. $$f(w) = \frac{1}{2n}\sum_{i=1}^n y_n - x_n^Tw$$, where $n$ is the number of sample in the training data.
With this scheme, the gradient is estimated by using only one row of our training data at each iteration. Note that in expectation, we also converge to a minimum.

- `logistic_regression(y, tx, initial_w, max_iters, gamma)`
This function uses the given training data, corresponding label and parameters needed (number of iterations, step-size and initial weights) to perform the gradient descent algorithm. It then computes the optimal weights by using the gradient descent update rule:
$$w_{k+1} = w_k - \gamma \nabla f(w)$$
where $w_k$ is the vector of weights at iteration $k$, $\gamma$ is the step-size and $f$ is the loss function. Here the loss function is the logistic loss function, i.e. $$f(w) = \sum_{i=1}^n \log{(1 + e^{-y_n\sigma(x_n^Tw)})}$$, where $n$ is the number of sample in the training data and $\sigma(\cdot)$ is the sigmoid function.
With this scheme, the gradient is estimated using all of our training data at each iteration.

- `reg_logistic_regression(y, tx, initial_w, max_iters, gamma, lambda_)`
This function uses the given training data, corresponding label, a regularization term $\lambda$ and parameters needed (number of iterations, step-size and initial weights) to perform the gradient descent algorithm. It then computes the optimal weights by using the gradient descent update rule:
$$w_{k+1} = w_k - \gamma \nabla f(w)$$
where $w_k$ is the vector of weights at iteration $k$, $\gamma$ is the step-size and $f$ is the loss function. Here the loss function is the logistic loss function with L2-regularization, i.e. $$f(w) = \sum_{i=1}^n \log{(1 + e^{-y_n\sigma(x_n^Tw)})} + \frac{\lambda}{2} \|w\|^2$$, where $n$ is the number of sample in the training data and $\sigma(\cdot)$ is the sigmoid function.
With this scheme, the gradient is estimated using all of our training data at each iteration.

### `features.py`
This contains all the methods needed to craft new features. 
It consists mainly of adding a polynomial basis (with certain degree) for each feature, and adding various other transformations such as $sin(\cdot), cos(\cdot), exp(\cdot)$, or even multiplying some of the columns and then applying those transformations.

- `add_features(input_data)` 
Given the input data, add all the features described above.

### `better_implementations.py`
This file contains the two added methods we chose to perform, that is the accelerated gradient scheme with adaptive restart. We did it both for the Least Squares loss function and the Logistic one.

- `accelerated_least_squares_GD(y, tx, initial_w, max_iters, lambda_)`
This function uses the given training data, corresponding label, a regularization term $\lambda$ and parameters needed (number of iterations and initial weights) to perform the accelerated gradient descent algorithm.
We use the fact that here, the loss function is $L$-Lipschitz with $L = \lambda_{max}(\textbf{X}^T \textbf{X}) + \lambda$, but also $\mu$-strongly convex with $\mu =  \lambda_{min}(\textbf{A}^T \textbf{A}) + \lambda$. (Here $\lambda_{max}(\cdot)$ and $\lambda_{min}(\cdot)$ represent the max and min eigenvalues.) It then computes the optimal weights by using the accelerated gradient descent update rule:
$$w_{k+1} = y_k - \frac{1}{L} \nabla f(y_k)$$
$$y_{k+1} = w_{k+1} + \frac{\sqrt{L} - \sqrt{\mu}}{\sqrt{L} + \sqrt{\mu}} (w_{k+1} - w_k)$$
where $w_k$ is the vector of weights at iteration $k$,  and $f$ is the loss function. Note that we set $y_0 = w_0$.
Here the loss function is the least squares loss function with L2-regularization, i.e. $$f(w) = \frac{1}{2n}\sum_{i=1}^n y_n - x_n^Tw + \frac{\lambda}{2} \|w\|^2$$, where $n$ is the number of sample in the training data.
Note that since the accelerated gradient scheme introduce an oscillatory behavior in the convergence to an optimum, we also added adaptive restart. That is, at each iteration $k$, if we have $\nabla f(y_k)^T (w_{k+1} -w_k) > 0$, we reset the momentum term.
With this scheme, the gradient is estimated using all of our training data at each iteration.

- `accelerated_reg_logistic_regression(y, tx, initial_w, max_iters, lambda_)`
This function uses the given training data, corresponding label, a regularization term $\lambda$ and parameters needed (number of iterations and initial weights) to perform the accelerated gradient descent algorithm.
We use the fact that here, the loss function is $L$-Lipschitz with $L = \frac{1}{4} \|\textbf{A}\| + \lambda$, but also $\mu$-strongly convex with $\mu =  \lambda$. It then computes the optimal weights by using the accelerated gradient descent update rule:
$$w_{k+1} = y_k - \frac{1}{L} \nabla f(y_k)$$
$$y_{k+1} = w_{k+1} + \frac{\sqrt{L} - \sqrt{\mu}}{\sqrt{L} + \sqrt{\mu}} (w_{k+1} - w_k)$$
where $w_k$ is the vector of weights at iteration $k$,  and $f$ is the loss function. Note that we set $y_0 = w_0$.
Here the loss function is the logistic loss function with L2-regularization, i.e. $$f(w) = \sum_{i=1}^n \log{(1 + e^{-y_n\sigma(x_n^Tw)})} + \frac{\lambda}{2} \|w\|^2$$, where $n$ is the number of sample in the training data and $\sigma(\cdot)$ is the sigmoid function.
Note that since the accelerated gradient scheme introduce an oscillatory behavior in the convergence to an optimum, we also added adaptive restart. That is, at each iteration $k$, if we have $\nabla f(y_k)^T (w_{k+1} -w_k) > 0$, we reset the momentum term.
With this scheme, the gradient is estimated using all of our training data at each iteration.

### `helpers.py`

This file contains all the small functions that make those in `implementations.py` work. It also contains the function that we used in order to preprocess the data.

- `standardize(x)`
Standardize the given input data (a matrix). It basically subtracts the mean of each columns for each column, and then normalize it by dividing by the standard devation.

- `compute_accuracy(y, tx, w)`
Computes the accuracy of a given model. It basically computes the labels of the input data `tx` by using the weights `w`, and then compare this prediction to the true labels `y`.

- `compute_loss_mse(y, tx, w, lambda_)`
Computes the value of the loss function (here being MSE), using the labels, input data and given weights. Note that we added the possibility to introduce regularization by accepting the `lambda_` parameter.

- `compute_gradient_mse(y, tx, w, lambda_)`
Computes the value of the gradient of the loss function (here being MSE), using the labels, input data and given weights. Note that we added the possibility to introduce regularization by accepting the `lambda_` parameter.

- `sigmoid(z)`
Compute the sigmoid function on input `z`.

- `compute_gradient_lg(x)`
Computes the value of the gradient of the loss function (here being Logistic loss), using the labels, input data and given weights. Note that we added the possibility to introduce regularization by accepting the `lambda_` parameter.

- `compute_loss_lg(x)`
Computes the value of the loss function (here being Logistic loss), using the labels, input data and given weights. Note that we added the possibility to introduce regularization by accepting the `lambda_` parameter.

- `batch_iter(y, tx, batch_size, num_batches, shuffle)`
Returns a mini-batch iterator on the input data `y, tx`. We can specify the batch, number of batches, and also have the option to shuffle our input data.

- `build_poly(x, degree)`
Computes a polynomial basis of input data `x` up to degree `degree`.

- `clean(input_data)`
Clean the input data by handling invalid values (here, -999). We chose to replace the invalid entries of a column by assigning them the most frequent valid value of that same column.

- `add_poly_features(input_data)`
Add polynomial basis for the given input data. We selected which columns need which degree for their polynomial basis based on a correlation factor (see report).

- `preprocess(input_data)`
Preprocess the input data by first cleaning invalid entries,  adding new features (polynomial basis, cos, ...), standardizing it, and finally adding the 1 vector as first column (for the bias terms).

### `cross_validation.py`
This file contains all the functions needed to perform cross-validation on the dataset.

- `build_k_indices(N, k-fold, seed)`
Builds k indices to perform k-fold cross validation. The k indices will be used to select the test data in our dataset.

- `cross_validation(y, x, k_indices, k, model, initial_w, max_iters, gamma, lambda_)`
Performs k-fold cross-validation on the dataset using the given model and its parameters.
We split the data according to the given k indices, use the given model to get the loss together with the optimal weights. 
We finally return also the accuracy on the training set and also the test set.

- `choose_model(y, tx, model, initial_w, max_iters, gamma, lambda_)`
Given input data and corresponding labels, it computes the optimal weights for the selected model.

### `Project1-ML.ipynb`
This notebook contains code that evaluate the different models and choose the best parameters by using cross-validation.
This was also useful to then report the accuracy of each models to determine which one performs the best.

### `run.py`
This file contains the code that loads the training and test data. 
It let the user choose which method to run.
We then find the optimal weights for the selected model and finally submit the predictions, in the format accepted by the Kaggle competition.

