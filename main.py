import itertools
import plotly
import plotly.graph_objs as go
import plotly.express as px
from more_itertools import powerset
from plotly.subplots import make_subplots
import numpy as np

from visualisation import Visualization

N = 1000
x = np.linspace(0, 1, N)
z = 20 * np.sin(2 * np.pi * 3 * x) + 100 * np.exp(x)
error = 10 * np.random.randn(N)
t = z + error

tr = 0.8
val = 0.1
ind_prm = np.random.permutation(np.arange(N))
train_ind = ind_prm[:int(tr * N)]
valid_ind = ind_prm[int(tr * N):int((val + tr) * N)]
test_ind = ind_prm[int((val + tr) * N):]
x_train, t_train, x_valid, t_valid, x_test, t_test = x[train_ind], t[train_ind], x[valid_ind], t[valid_ind], x[
    test_ind], t[test_ind]

functions = [lambda x: np.sin(x), lambda x: np.cos(x), lambda x: np.log(x + 1e-7), lambda x: np.exp(x),
             lambda x: np.sqrt(x), lambda x: x, lambda x: x ** 2, lambda x: x ** 3]
indexes = [0, 1, 2, 3, 4, 5, 6, 7]
func_names = ["sin(x)", "cos(x)", "log(x + 1e-7)", "exp(x)", "sqrt(x)", "x", "x^2", "x^3"]

sets = list(powerset(indexes))[1:93]
namess = list(powerset(func_names))[1:93]


def matrix_F(x, ind):
    F = np.ones((1, len(x)))
    for i in ind:
        F = np.append(F, [functions[i](x)], axis=0)
    return F.T


def learn(F, t):
    return ((np.linalg.pinv(F.T.dot(F))).dot(F.T)).dot(t)


def error(W, t, x, ind):
    F = matrix_F(x, ind)
    return (1 / 2) * sum((W.dot(F.T)) - t) ** 2


def toFixed(numObj, digits=0):
    return f"{numObj:.{digits}f}"


error_test = []
error_valid = []
for i in sets:
    F = matrix_F(x_train, i)
    W = learn(F, t_train)
    error_valid.append(error(W, t_valid, x_valid, i))
    error_test.append(error(W, t_test, x_test, i))
min_error_valid = np.array(sorted(error_valid)[:10])
min_errors_index = [error_valid.index(i) for i in min_error_valid]
min_error_test = np.array([error_test[i] for i in min_errors_index])

names = []
for i in min_errors_index:
    W = learn(matrix_F(x_train, sets[i]), t_train)
    str = f"y = {toFixed(W[0], 2)}"
    for j in range(1, len(sets[i]) + 1):
        if W[j] > 0:
            str += f" + {toFixed(W[j], 2)}{func_names[sets[i][j - 1]]}"
        else:
            str += f" - {toFixed(abs(W[j]), 2)}{func_names[sets[i][j - 1]]}"
    names.append(str)

visualisation = Visualization()
visualisation.models_error_scatter_plot(min_error_valid, min_error_test, np.array(names), 'title', show=True, save=True,
                                        name="ML_HW2",
                                        path2save="C:/Users/26067/PycharmProjects/ML_HW2")
