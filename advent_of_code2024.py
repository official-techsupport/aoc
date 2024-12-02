#!/usr/bin/env python3

import advent_of_code_utils
from advent_of_code_utils import *
utils_init(2024, globals())

###########


def problem1(data, second):
    _data = split_data('''
3   4
4   3
2   5
1   3
3   9
3   3
 ''')
    lst1, lst2 = [], []
    for s in data:
        a, b = s.split()
        lst1.append(int(a))
        lst2.append(int(b))
    lst1.sort()
    lst2.sort()
    if not second:
        return sum(abs(a - b) for a, b in zip(lst1, lst2))
    lst2 = Counter(lst2)
    return sum(x * lst2[x] for x in lst1)


def problem2(data, second):
    _data = split_data('''
7 6 4 2 1
1 2 7 8 9
9 7 6 2 1
1 3 2 4 5
8 6 4 4 1
1 3 6 7 9
    ''')
    def is_safe(r):
        pairs = list(pairwise(r))
        return all(1 <= abs(a - b) <= 3 for a, b in pairs) and (
            all(a < b for a, b in pairs) or
            all(a > b for a, b in pairs))

    def is_safe2(r):
        if is_safe(r):
            return True
        for i in range(len(r)):
            r2 = list(r)
            del r2[i]
            if is_safe(r2):
                return True
        return False

    rs = [[int(c) for c in s.split()] for s in data]
    if second:
        return sum(is_safe2(r) for r in rs)
    return sum(is_safe(r) for r in rs)


def problem3(data, second):
    data = 'xmul(2,4)%&mul[3,7]!@^do_not_mul(5,5)+mul(32,64]then(mul(11,8)mul(8,5))'
    data = '''xmul(2,4)&mul[3,7]!^don't()_mul(5,5)+mul(32,64](mul(11,8)undo()?mul(8,5))'''
    data = get_raw_data()
    rx = re.compile(r'mul\((\d+),(\d+)\)|do(n\'t)?\(\)')
    lst = rx.findall(data)
    res = 0
    enabled = True
    for a, b, x in lst:
        # print(a, b, x)
        if not a:
            if second:
                enabled = not x
        elif enabled:
            res += int(a) * int(b)
    return res


#########

def problem(data, second):
    data = split_data(''' ''')
    if second: return

    return None


##########

if __name__ == '__main__':
    print('Hello')
    solve_latest()
    # solve_all()
