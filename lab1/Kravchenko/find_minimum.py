import math
from random import uniform


def f(list):
    return (1 - list[0]) ** 2 + 100 * (list[1] - list[0] ** 2) ** 2


def grad_f(list):
    return [-2 * (1 - list[0]) - 400 * (list[1] - list[0] ** 2) * list[0], 200 * (list[1] - list[0] ** 2)]


def norm(list):
    ans = 0
    for i in range(len(list)):
        ans += i ** 2
    return math.sqrt(ans)


def optimal_lambda(list, n, lamb):
    my_list = []
    const = float('inf')
    for i in range(0, n):
        lamb = lamb * 0.6
        for j in range(0, len(list)):
            my_list.append(list[j] - lamb * grad_f(list)[j])
        if f(my_list) < const:
            const = f(my_list)
    return lamb


def gradient_descent(list0, lamb, eps, type):
    list1 = []
    for i in range(len(list0)):
        list1.append(list0[i] - lamb * grad_f(list0)[i])
    while True:
        if type == 2:
            lamb *= 0.5
        elif type == 3:
            lamb = optimal_lambda(list1, 10, lamb)
        flag = True
        for i in range(len(list0)):
            if abs(list1[i] - list0[i]) >= eps:
                flag = False
        if flag:
            break
        if abs(f(list1) - f(list0)) < eps:
            break
        if norm(grad_f(list1)) < eps:
            break
        for i in range(len(list0)):
            list0[i] = list1[i]
        for i in range(len(list0)):
            list1[i] = list0[i] - lamb * grad_f(list0)[i]
    return f(list1)


def monte_carlo(n, m):
    list = []
    for i in range(m):
        list.append(uniform(-10, 10))
    local_minimum = f(list)
    list_min = list
    for i in range(n):
        list_r = []
        for i in range(m):
            list_r.append(uniform(-3, 3))
        res = gradient_descent(list_r, 0.0001, 0.001, 3)
        if res < local_minimum:
            list_min = list_r
            local_minimum = res
    return local_minimum, list_min


if __name__ == '__main__':
    minimum = monte_carlo(100, 2)
    for i in range(10):
        y = monte_carlo(100, 2)
        if y < minimum:
            minimum = y
    print(minimum)
