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


def problem4(data, second):
    # if second: return
    _data = split_data('''
MMMSXXMASM
MSAMXMSMSA
AMXSXMAAMM
MSAMASMSMX
XMASAMXAMM
XXAMMXXAMA
SMSMSASXSS
SAXAMASAAA
MAMMMXMMMM
MXMXAXMASX''')
    height = len(data)
    width = len(data[0])
    assert all(len(row) == width for row in data)

    def get(pos):
        row, col = pos
        if 0 <= row < height and 0 <= col < width:
            return data[row][col]
        return ''

    visited = set()
    query = 'XMAS'
    def rec(pos, d, idx, path=[]):
        if idx >= len(query):
            visited.update(path)
            return True
        path = path + [pos]
        if get(pos) != query[idx]:
            return False
        return rec(addv2(pos, d), d, idx + 1, path)

    occurs = 0
    if not second:
        for row in range(height):
            for col in range(width):
                for d in directions8:
                    occurs += rec((row, col), d, 0)
        return occurs

    query = 'MAS'
    for row in range(height):
        for col in range(width):
            pos = (row, col)
            found = 0
            for d in ((-1, -1), (1, 1)):
                found += rec(addv2(pos, d), mulv2s(d, -1), 0)
            if not found:
                continue
            found = 0
            for d in ((-1, 1), (1, -1)):
                found += rec(addv2(pos, d), mulv2s(d, -1), 0)
            if found:
                occurs += 1
    return occurs

    # data2 = [list(s) for s in data]
    # for row in range(height):
    #     for col in range(width):
    #         if (row, col) not in visited:
    #             data2[row][col] = ' '
    # print('\n'.join(''.join(row) for row in data2))


def problem5(data, second):
    _data = split_data('''
47|53
97|13
97|61
97|47
75|29
61|13
75|53
29|13
97|29
53|29
61|53
97|53
61|29
47|13
75|47
97|75
47|61
75|61
47|29
75|13
53|13

75,47,61,53,29
97,61,53,29,13
75,29,13
75,97,47,61,53
61,13,29
97,13,75,29,47
''')
    before = defaultdict(list)
    data_it = iter(data)
    for line in data_it:
        if not line:
            break
        a, b = line.split('|')
        before[int(b)].append(int(a))

    res = []
    for line in data_it:
        pgs = [int(p) for p in line.split(',')]
        good = True
        for i, p in enumerate(pgs):
            bef = before[p]
            for p2 in pgs[i + 1 : ]:
                if p2 in bef:
                    good = False
                    break
            if not good:
                break
        if good:
            if not second:
                res.append(pgs)
            continue
        if not second:
            continue
        while True:
            fixed = False
            for i, p in enumerate(pgs):
                bef = before[p]
                for j, p2 in enumerate(pgs[i + 1 : ]):
                    if p2 in bef:
                        assert p2 == pgs.pop(i + 1 + j)
                        pgs.insert(i, p2)
                        fixed = True
                        break
                if fixed:
                    break
            if not fixed:
                break
        res.append(pgs)

    return sum(pgs[len(pgs)//2] for pgs in res)








#########

def problem(data, second):
    if second: return
    data = split_data(''' ''')


##########

if __name__ == '__main__':
    print('Hello')
    solve_latest()
    # solve_all()
