
import matplotlib.pyplot as plt
import numpy as np

from proj1_helpers import *

y_tr, x_tr, id_tr = load_csv_data("../../train.csv")
y_te, x_te, id_te = load_csv_data("../../test.csv")
'''
x_tr_b = x_tr[x_tr[1] == 'b']
x_tr_s = x_tr[x_tr[1] == 's']
'''
means_tr = np.mean(x_tr, axis=0)
stdev_tr = np.std(x_tr, axis=0)
'''
means_tr_b = np.mean(x_tr_b, axis=0)
stdev_tr_b = np.std(x_tr_b, axis=0)
means_tr_s = np.mean(x_tr_s, axis=0)
stdev_tr_s = np.std(x_tr_s, axis=0)
'''
idx = np.arange(means_tr.shape[0])

plt.errorbar(idx, means_tr, stdev_tr, linestyle='None', marker='^')

plt.show()
'''
waitkey(0)

plt.errorbar(idx, means_tr_b, stdev_tr_b, linestyle='None', marker='^')

plt.show()
waitkey(0)

plt.errorbar(idx, means_tr_s, stdev_tr_s, linestyle='None', marker='^')

plt.show()
'''
# 30 parameters -> 435 correlation tests

# mean + variance
# correlation of parameters

