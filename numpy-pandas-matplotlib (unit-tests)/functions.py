from typing import List


def sum_non_neg_diag(X: List[List[int]]) -> int:
    sum = -1
    n, m = len(X), len(X[0])
    for i in range(min(n, m)) :
        if X[i][i] >= 0:
            if sum == -1:
                sum = 0
            sum += X[i][i]
    return sum


def are_multisets_equal(x: List[int], y: List[int]) -> bool:
    if len(x) != len(y) :
        return False
    f = []
    for i in y :
        f.append(i)
    for i in x :
        if not i in f:
            return False
        f.remove(i)
    return True


def max_prod_mod_3(x: List[int]) -> int:
    maxx = None
    for i in range(len(x) - 1) :
        a = x[i] * x[i + 1]
        if (not(a % 3)) :
            if not maxx :
                maxx = a
            if a > maxx :
                maxx = a
    return maxx if maxx else -1


def convert_image(image: List[List[List[float]]], weights: List[float]) -> List[List[float]]:
    for i in range(len(weights)) :
        for j in image:
            for y in j:
                y[i] *= weights[i]
    summa = []
    for i in image :
        tmp_summa =[]
        for j in i :
            tmp_summa.append(sum(j))
        summa.append(tmp_summa)
    return summa


def rle_scalar(x: List[List[int]], y:  List[List[int]]) -> int:
    res = []
    n = sum([x[i][1] for i in range(len(x))])
    m = sum([y[i][1] for i in range(len(y))])
    if n != m :
        return -1
    x1, y1 = [], []
    for i in x :
        for j in range(i[1]) :
            x1.append(i[0])
    for i in y :
        for j in range(i[1]) :
            y1.append(i[0])
    for i in range(n) :
        res.append(x1[i] * y1[i])
    return sum(res)


def cosine_distance(X: List[List[float]], Y: List[List[float]]) -> List[List[float]]:
    n, m = len(X), len(Y)
    res = [[0]*m for i in range(n)]
    for i in range(n) : 
        for j in range(m) :
            flag_x, flag_y = True, True
            for k in range(len(X[i])) :
                if X[i][k] != 0 :
                    flag_x = False
                if Y[j][k] != 0 :
                    flag_y = False
            if any((flag_x, flag_y)) :
                res[i][j] = 1
            else :

                def skal(a, b) :
                    sum = 0
                    for i in range(len(a)) :
                        sum += a[i] * b[i]
                    return sum
                
                def metr(a) :
                    sum = 0
                    for i in a:
                        sum += i**2
                    return sum**0.5
                
                res[i][j] = skal(X[i], Y[j]) / (metr(X[i]) * metr(Y[j]))
    return res