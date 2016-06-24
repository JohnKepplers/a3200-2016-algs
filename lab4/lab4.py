import math
from sklearn import datasets
import numpy
import random
import sklearn.ensemble as sk
from sklearn.cross_validation import cross_val_score


class My_CaRT():
    def __init__(self):
        self.x = []
        self.y = []
        self.number_of_trees = 5

    def gini(self, set):
        categories = []
        squares = 0
        sum = 0
        for i in range(3):
            categories += [0]
        for i in range(len(set)):
            categories[set[i]] += 1
        for i in range(len(categories)):
            sum += categories[i]
        for i in range(len(categories)):
            squares += (categories[i] / sum) ** 2
        return 1 - squares

    def entropy(self, set):
        categories = []
        sum = 0
        res = 0
        for i in range(3):
            categories += [0]
        for i in range(len(set)):
            categories[set[i]] += 1
        for i in range(len(categories)):
            sum += categories[i]
        for i in range(len(categories)):
            if categories[i] == 0:
                continue
            else:
                res += categories[i] / sum * math.log2(categories[i] / sum)
        return -res

    def data_processing(self):
        iris = datasets.load_iris()
        self.x = iris.data
        self.y = iris.target

    def gain(self, set, set1, set2, type):
        if type == "entropy":
            return self.entropy(set) - (len(set1) / len(set)) * self.entropy(set1) - (len(set2) / len(
                set)) * self.entropy(set2)
        elif type == "gini":
            return (len(set1) / len(set)) * self.gini(set1) - (len(set2) / len(set)) * self.gini(set2)

    def CaRT(self, x, y):
        c = 0
        true_i = 0
        true_j = 0
        for i in range(3):
            for j in range(len(x)):
                y1_test = []
                y2_test = []
                for k in range(0, len(x)):
                    if x[k][i] >= x[j][i]:
                        y1_test = y1_test + [y[k]]
                    if x[k][i] < x[j][i]:
                        y2_test = y2_test + [y[k]]
                if (len(y1_test) != 0) and (len(y2_test) != 0) and ((self.gain(y, y1_test, y2_test, "gini")) > c):
                    true_i = i
                    true_j = j
                    c = self.gain(y, y1_test, y2_test, "gini")
        flag = True
        for i in range(len(x) - 1):
            if x[i][true_i] != x[i + 1][true_i]:
                flag = False
        if flag:
            return False
        else:
            return true_i, true_j

    def my_tree(self, x, y, v):
        if not self.CaRT(x, y):
            return y[0]
        else:
            y1_index = []
            y2_index = []
            y_left = []
            y_right = []
            true_j, true_i = self.CaRT(x, y)
            for i in range(len(x)):
                if x[i][true_j] >= x[true_i][true_j]:
                    y1_index = y1_index + [i]
                    y_left = y_left + [y[i]]
                if x[i][true_j] < x[true_i][true_j]:
                    y2_index = y2_index + [i]
                    y_right = y_right + [y[i]]
            l_index_0 = y1_index[0]
            x_left = x[l_index_0]
            for i in range(1, len(y1_index)):
                x_left = numpy.vstack((x_left, x[y1_index[i]]))
            r_index_0 = y2_index[0]
            x_right = x[r_index_0]
            for i in range(1, len(y2_index)):
                x_right = numpy.vstack((x_right, x[y2_index[i]]))
            counter_left = True
            for i in range(0, len(y_left) - 1):
                if y_left[i] != y_left[i + 1]:
                    counter_left = False
            counter_right = True
            for i in range(0, len(y_right) - 1):
                if y_right[i] != y_right[i + 1]:
                    counter_right = False
            if (v[true_j] >= x[true_i][true_j]) and (counter_left == True):
                return y_left[0]
            elif (v[true_j] < x[true_i][true_j]) and (counter_right == True):
                return y_right[0]
            elif (v[true_j] >= x[true_i][true_j]) and (counter_left != True):
                return self.my_tree(x_left, y_left, v)
            elif (v[true_j] < x[true_i][true_j]) and (counter_right != True):
                return self.my_tree(x_right, y_right, v)

    def cross_validation_for_CaRT(self, x, y):
        error = 0
        k = numpy.shape(x)[0]
        for j in range(1, k - 1):
            X_matrix = numpy.vstack((x[0:j], x[j + 1: k]))
            y_matrix = numpy.hstack((y[0:j], y[j + 1: k]))
            t = self.my_tree(X_matrix, y_matrix, x[j])
            if y[j] != t:
                error = error + 1
            print(error, k)
        return ((k - 2) - error) / (k - 2)

    def random_forest(self, x, y, n, m, v, k):
        a, b, c = 0, 0, 0
        for t in range(k):
            z = random.randint(0, len(x) - 1)
            y1 = [y[z]]
            x1 = x[z]
            for i in range(0, n - 1):
                z = random.randint(0, len(x) - 1)
                x1 = numpy.vstack((x1, x[z]))
                y1 = y1 + [y[z]]
            a = random.randint(0, numpy.shape(x)[1] - 1)
            x2 = x1[:, a]
            v1 = [v[a]]
            for i in range(0, m - 1):
                a = random.randint(0, numpy.shape(x)[1] - 1)
                x2 = numpy.vstack((x2, x1[:, a]))
                v1 = v1 + [v[a]]
            x2 = x2.transpose()
            answer = self.my_tree(x2, y1, v1)
            if answer == 0:
                a += 1
            elif answer == 1:
                b += 1
            elif answer == 2:
                c += 2
            if max(a, b, c) == a:
                return 0
            elif max(a, b, c) == b:
                return 1
            else:
                return 2

    def cross_validation_for_random_forest(self, x, y, m, z):
        error = 0
        k = numpy.shape(x)[0]
        for j in range(1, k - 1):
            X_matrix = numpy.vstack((x[0:j], x[j + 1: k]))
            y_matrix = numpy.hstack((y[0:j], y[j + 1: k]))
            t = self.random_forest(X_matrix, y_matrix, 150, m, x[j], z)
            if y[j] != t:
                error = error + 1
        return ((k - 2) - error) / (k - 2)

    def grid_search(self, x, y):
        M = 0
        Z = 0
        cv = 0
        for z in range(0, 20):
            a = self.cross_validation_for_random_forest(x, y, 4, z)
            print(a)
            if a > cv:
                cv = a
                Z = z
        cv = 0
        for m in range(0, 4):
            a = self.cross_validation_for_random_forest(x, y, m, Z)
            if a > cv:
                cv = a
                M = m
        return M, Z


if __name__ == '__main__':
    CaRT = My_CaRT()
    CaRT.data_processing()
    random_forest = sk.RandomForestClassifier()
    scores = cross_val_score(random_forest, CaRT.x, CaRT.y)
    print(scores.mean())
    print(CaRT.cross_validation_for_random_forest(CaRT.x, CaRT.y, 4, 5))
    print(CaRT.grid_search(CaRT.x, CaRT.y))
