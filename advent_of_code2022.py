#!/usr/bin/env python3

from advent_of_code_utils import *
utils_init(2022, globals())

###########


def problem1(data, second):
    data = get_raw_data()
    data = data.split('\n\n')
    lst_lst = [s.split() for s in data]
    sums = [sum(int(s) for s in lst) for lst in lst_lst]
    if not second:
        return max(sums)
    sums = [-it for it in sums]
    heapify(sums)
    return -sum(heappop(sums) for _ in range(3))


def problem2(data, second):
#     data = split_data('''A Y
# B X
# C Z''')
    moves = [s.split() for s in data]
    moves = [('ABC'.index(p1), 'XYZ'.index(p2)) for p1, p2 in moves]
    res = 0
    for p1, p2 in moves:
        if not second:
            res += 1 + p2
            res += [3, 6, 0][(p2 - p1) % 3]
        else:
            res += [(p1 - 1) % 3, p1, (p1 + 1) % 3][p2] + 1
            res += [0, 3, 6][p2]

    return res



##########

def problem(data, second):
    if second: return
    return None

##########

if __name__ == '__main__':
    print('Hello')
    # solve_all()
    # solve_latest(22)
    solve_latest()
