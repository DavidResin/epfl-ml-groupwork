from proj1_helpers import *
from utility import *
from implementations import *

y_tr, x_tr, ids_tr = load_csv_data("../train.csv")
y_te, x_te, ids_te = load_csv_data("../test.csv")

degree = 7
x_trained = build_poly(x_tr, degree)
x_tested = build_poly(x_te, degree)

w, loss = least_squares(y_tr, x_trained)

y_pred = predict_labels(w, x_tested)

create_csv_submission(ids_te, y_pred, 'submit.csv')
