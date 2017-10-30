from proj1_helpers import *
from implementations import *

y_tr, x_tr, ids_tr = load_csv_data("../../train.csv")
y_te, x_te, ids_te = load_csv_data("../../test.csv")

triage(x_tr)
triage(x_te)

# Use degree to create polynomials of given degree.
degree = 9
lambda_ = 2.6826957952797275e-09
iters = 1000
gamma = 0.01
x_trained = build_poly(x_tr, degree)
x_tested = build_poly(x_te, degree)

w = np.zeros(x_trained.shape[1])
w, loss = reg_logistic_regression(y_tr, x_trained,lambda_, w, iters, gamma)

# Get predictions.
y_pred = predict_labels(w, x_tested)
print(y_pred)
# Create the csv from the obtained predictions.
create_csv_submission(ids_te, y_pred, 'submit.csv')
