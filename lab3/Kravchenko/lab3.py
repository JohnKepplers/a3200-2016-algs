import math
from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier


class My_kNN():
    def __init__(self):
        self.known_x = []
        self.known_y = []
        self.unknown_x = []
        self.unknown_y = []
        self.is_weight = False
        self.k = 11

    def distance(self, m, x, y):
        if m == 0:
            return math.sqrt((x[0] - y[0]) ** 2 + (x[1] - y[1]) ** 2)
        elif m == 1:
            return math.fabs(x[0] - y[0]) + math.fabs(x[1] - y[1])
        elif m == 2:
            if math.fabs(x[0] - y[0]) > math.fabs(x[1] - y[1]):
                return math.fabs(x[0] - y[0])
            else:
                return math.fabs(x[1] - y[1])

    def data_processing(self):
        iris = datasets.load_iris()
        x = iris.data
        y = iris.target
        for i in range(len(x)):
            if 40 <= i < 50 or 90 <= i < 100 or 140 <= i < 150:
                self.known_x += [[x[i][2], x[i][3]]]
                self.known_y += [y[i]]
            else:
                self.unknown_x += [[x[i][2], x[i][3]]]
                self.unknown_y += [y[i]]
        return self.kNN()

    def kNN(self):
        result = []
        for i in range(len(self.unknown_x)):
            array_of_neighbours = []
            array_of_distances = []
            array_of_weight = [0, 0, 0]
            for j in range(len(self.known_y)):
                array_of_neighbours.append(self.known_y[j])
                array_of_distances.append(self.distance(0, self.unknown_x[i], self.known_x[j]))
            while len(array_of_neighbours) > self.k:
                del array_of_neighbours[array_of_distances.index(max(array_of_distances))]
                del array_of_distances[array_of_distances.index(max(array_of_distances))]
            if self.is_weight:
                for i in range(len(array_of_distances)):
                    if array_of_distances[i] != 0:
                        array_of_weight[array_of_neighbours[i]] += 1 / array_of_distances[i]
                result.append(array_of_weight.index(max(array_of_weight)))
            else:
                a, b, c = 0, 0, 0
                for j in range(len(array_of_neighbours)):
                    if array_of_neighbours[j] == 0:
                        a += 1
                    elif array_of_neighbours[j] == 1:
                        b += 1
                    else:
                        c += 1
                if max(a, b, c) == a:
                    result.append(0)
                elif max(a, b, c) == b:
                    result.append(1)
                else:
                    result.append(2)
        return result

    def enemy_kNN(self):
        iris = datasets.load_iris()
        x = iris.data
        y = iris.target
        kNN = KNeighborsClassifier(n_neighbors=3)
        kNN.fit(x, y)
        return kNN.score(x, y)

    def cross_validation(self):
        mistakes = 0
        result = self.data_processing()
        if self.unknown_y[0] != result[0]:
            mistakes += 1
        for i in range(1, len(self.unknown_y)):
            result = self.kNN()
            if self.unknown_y[i] != result[i]:
                mistakes += 1
        return (len(self.unknown_y) - mistakes) / len(self.unknown_y)

    def grid_search(self):
        self.k = 0
        this_k = self.k
        min_mistake = self.cross_validation()
        print("1 /30")
        for i in range(1, 30):
            print(i + 1, "/30")
            self.k = i
            mis = self.cross_validation()
            print(mis)
            if mis < min_mistake:
                min_mistake = mis
                this_k = self.k
        return this_k


if __name__ == '__main__':
    my_kNN = My_kNN()
    print(my_kNN.enemy_kNN())
    print(my_kNN.cross_validation())
