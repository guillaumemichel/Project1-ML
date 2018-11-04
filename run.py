from better_implementations import *
from proj1_helpers import *
from helpers import preprocess, compute_accuracy
from cross_validation import cross_validation, choose_model
import matplotlib.pyplot as plt

models = ["least_squares",
        "least_squares_GD",
        "least_squares_SGD",
        "ridge_regression",
        "logistic_regression",
        "reg_logistic_regression",
        "accelerated_least_squares_GD",
        "accelerated_reg_logistic_regression"]

print("###### ML Project 1 ######\n")
print("Enter a number to choose one of the following methods: ")

for i, model in enumerate(models):
    print(i, "->", model)

model = -1
while model < 0 or model > 7:
    model = input("Enter a valid number: ")
    try:
        model = int(model)
    except ValueError:
        model = -1

#Load the train and test data
print("Loading the data...")

y_tr, input_data_train, _ = load_csv_data("data/train.csv")
y_te, input_data_test, ids_test = load_csv_data("data/test.csv")

#Preprocess train and test data
print("Preprocessing the data...")

tx_tr = preprocess(input_data_train)
tx_te = preprocess(input_data_test)

#Compute the optimal weights
print("Computing the optimal weights...")

losses, optimal_weights = choose_model(y_tr, tx_tr, models[model], np.zeros(tx_tr.shape[1]), 500, 2e-6, 0.0008)

print("Test accuracy: ", compute_accuracy(y_te, tx_te, optimal_weights))
print("Training accuracy: ", compute_accuracy(y_tr, tx_tr, optimal_weights))

y_pred = predict_labels(optimal_weights, tx_te)
create_csv_submission(ids_test, y_pred, "submission.csv")
