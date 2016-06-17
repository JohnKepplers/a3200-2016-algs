from random import randint


def function1(x):
    return x[0] and (x[9] or not x[8]) and (x[1] or not x[2] or not x[3]) or x[7] and x[4] and x[5] and x[6]


class Genetic_algorithm():
    def __init__(self, m, n):
        self.const = n
        self.n = n
        self.m = m
        self.population = []
        self.function = function1
        self.flag = False

    def create_population(self):
        for i in range(self.n):
            self.population += [randint(512, 1023)]

    def crutch(self, object):
        my_list = list(bin(object))[2:]
        for i in range(len(my_list)):
            my_list[i] = bool(int(my_list[i]))
        return my_list

    def fit(self):
        good = 0
        bad = 0
        for i in range(self.n):
            if self.function(self.crutch(self.population[i])) == False:
                good += 1
            else:
                bad += 1
        if good / (bad + good) > 0.9:
            self.flag = True

    def crossing_over(self, object1, object2):
        t_str = ""
        o1 = self.crutch(object1)
        o2 = self.crutch(object2)
        mask = self.crutch(self.population[randint(0, self.n - 1)])
        child = []
        for i in range(len(mask)):
            if mask[i]:
                child.append(o1[i])
            else:
                child.append(o2[i])
        for i in range(len(child)):
            child[i] = int(child[i])
            t_str += str(child[i])
        child = int(t_str, 2)
        self.n += 1
        return child

    def create_new_generation(self, number):
        if number % 2 == 1:
            number += 1
        cool_objects = []
        for i in range(number):
            object_one = self.population[randint(0, self.n - 1)]
            object_two = self.population[randint(0, self.n - 1)]
            if self.function(self.crutch(object_one)) < self.function(self.crutch(object_two)):
                cool_objects.append(object_one)
            else:
                cool_objects.append(object_two)
        while len(cool_objects) != 0:
            i = randint(0, len(cool_objects) - 1)
            object1 = cool_objects[i]
            del cool_objects[i]
            i = randint(0, len(cool_objects) - 1)
            object2 = cool_objects[i]
            del cool_objects[i]
            self.population.append(self.crossing_over(object1, object2))

    def selection(self):
        k = self.n
        iterator = 0
        new_population = []
        for i in range(self.n):
            if self.function(self.crutch(self.population[i])) == False:
                new_population.append(self.population[i])
        if len(new_population) == self.const:
            self.population = new_population
            self.n = self.const
        elif len(new_population) > self.const:
            while len(new_population) != self.const:
                i = randint(0, len(new_population) - 1)
                del new_population[i]
            self.population = new_population
            self.n = self.const
        elif len(new_population) < self.const:
            while len(new_population) != self.const:
                new_population.append(self.population[randint(0, len(self.population) - 1)])
            self.population = new_population
            self.n = self.const

        while self.n != self.const:
            iterator += 1
            i = randint(0, self.n - 1)
            if self.function(self.crutch(self.population[i])) == True:
                del self.population[i]
                self.n -= 1
            if iterator > 50:
                while self.n != self.const:
                    print(self.n, self.const)
                    j = randint(0, self.n - 1)
                    del self.population[j]
                    self.n -= 1


GA = Genetic_algorithm(100, 200, 200)
GA.create_population()
while not (GA.flag):
    GA.fit()
    GA.create_new_generation(70)
    GA.selection()
local_minimum = GA.population[0]
if GA.function(GA.crutch(local_minimum)) == False:
    my_list = list(bin(local_minimum))[2:]
    for i in range(len(my_list)):
        if my_list[i] == '1':
            my_list[i] = True
        elif my_list[i] == '0':
            my_list[i] = False
    print(my_list)
else:
    i = 1
    while local_minimum:
        local_minimum = GA.population[i]
        i += 1
    my_list = list(bin(local_minimum))[2:]
    for i in range(len(my_list)):
        if my_list[i] == '1':
            my_list[i] = True
        elif my_list[i] == '0':
            my_list[i] = False
    print(my_list)
