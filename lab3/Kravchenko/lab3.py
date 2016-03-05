from sklearn import datasets
import math


def distance(m, x, y, v):
    if m == 0:
        return math.sqrt((x[0] - y[0]) ** 2 + (x[1] - y[1]) ** 2), v
    elif m == 1:
        return math.fabs(x[0] - y[0]) + math.fabs(x[1] - y[1]), v
    elif m == 2:
        if math.fabs(x[0] - y[0]) > math.fabs(x[1] - y[1]):
            return math.fabs(x[0] - y[0]), v
        else:
            return math.fabs(x[1] - y[1]), v


def data_processing():
    dict_learning = {}
    dict_test = {}
    iris = datasets.load_iris()
    x = iris.data
    y = iris.target
    for i in range(len(x)):
        xx = [(x[i][2]), x[i][3]]
        if 40 <= i < 50 or 90 <= i < 100 or 140 <= i < 150:
            dict_test.update({frozenset(xx): [y[i], None]})
        else:
            dict_learning.update({frozenset(xx): y[i]})
    return kNN(20, dict_learning, dict_test, 50)


def kNN(k, dict_learning, dict_test, weight):
    for k_test in dict_test.keys():
        helping_array = []
        array_of_distances = []
        d = {}
        for k_learn, v_learn in dict_learning.items():
            dist, v = distance(0, list(k_test), list(k_learn), v_learn)
            d.update({dist: v})
            helping_array.append(dist)
        helping_array.sort()
        a, b, c = 0, 0, 0
        for i in range(k):
            array_of_distances.append(helping_array[i])
            if weight == 0:
                if d[array_of_distances[i]] == 0:
                    a += 1
                elif d[array_of_distances[i]] == 1:
                    b += 1
                else:
                    c += 1
            else:
                if d[array_of_distances[i]] == 0:
                    a += weight - i
                elif d[array_of_distances[i]] == 1:
                    b += weight - i
                else:
                    c += weight - i
        d = 0
        if max(a, b, c) == b:
            d += 1
        elif max(a, b, c) == c:
            d += 2
        dict_test[k_test][1] = d
    return dict_test


if __name__ == '__main__':
    print(data_processing())
