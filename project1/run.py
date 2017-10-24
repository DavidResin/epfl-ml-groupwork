import os, sys

sys.path.append(os.path.join(sys.path[0], 'scripts'))

from proj1_helpers import *
from implementations import *

y_tr, x_tr, ids_tr = load_csv_data("../train.csv")
y_te, x_te, ids_te = load_csv_data("../test.csv")

triage(x_tr)
triage(x_te)

# Use degree to create polynomials of given degree.
degree = 9
x_trained = build_poly(x_tr, degree)
x_tested = build_poly(x_te, degree)

# Use least squares to the w for predictions.
w, loss = least_squares(y_tr, x_trained)
#w, loss = logistic_regression(y_tr, x_trained, w, 10, 0.01)

# Get predictions.
y_pred = predict_labels(w, x_tested)

# Create the csv from the obtained predictions.
create_csv_submission(ids_te, y_pred, 'submit.csv')
