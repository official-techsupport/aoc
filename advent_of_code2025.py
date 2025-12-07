#!/usr/bin/env python3

import advent_of_code_utils
from advent_of_code_utils import *
utils_init(2025, globals())

###########


def problem1(data, second):
    # if second: return
    _data = split_data('''
L68
L30
R48
L5
R60
L55
L1
L99
R14
L82
''')
    cur = 50
    zeroes = 0
    for s in data:
        s = s.replace('L', '-').replace('R', '')
        offset = int(s)
        assert offset
        if second:
            new = cur + offset
            if new <= 0:
                zeroes += abs(new) // 100 + (cur != 0)
            else:
                zeroes += new // 100
            # print(f'{cur=} {offset=} {new=} {zeroes=}')
            cur = new % 100
        else:
            cur = (cur + offset) % 100
            if cur == 0:
                zeroes += 1
    return zeroes


def problem2(data, second):
    # if second: return
    _data = split_data('''
11-22,95-115,998-1012,1188511880-1188511890,222220-222224,
1698522-1698528,446443-446449,38593856-38593862,565653-565659,
824824821-824824827,2121212118-2121212124
''')
    data = ''.join(data).split(',')
    ranges = []
    total = 0
    for s in data:
        a, b = map(int, s.split('-'))
        ranges.append((a, b + 1))
        total += b - a + 1
    # print(total)
    total = 0
    if second:
        def is_invalid(i):
            s = str(i)
            for l in range(1, len(s) // 2 + 1):
                d, m = divmod(len(s), l)
                if m: continue
                x = s[:l]
                subranges = (s[i * l : (i + 1) * l] for i in range(1, d))
                if all(x == r for r in subranges):
                    # print(i)
                    return i
    else:
        def is_invalid(i):
            s = str(i)
            d, m = divmod(len(s), 2)
            if m: return
            if s[:d] == s[d:]:
                return i

    for a, b in ranges:
        for i in range(a, b):
            r = is_invalid(i)
            if r:
                total += r
    return total


def problem3(data, second):
    # if second: return
    _data = split_data('''
987654321111111
811111111111119
234234234234278
818181911112111
''')
    if not second:
        total = 0
        for s in data:
            m = 0
            for i, c1 in enumerate(s):
                for c2 in s[i + 1:]:
                    j = int(c1 + c2)
                    m = max(m, j)
            total += m
        return total
    # second
    def solve(s):
        @functools.cache
        def recur(pos, remaining):
            if not remaining:
                return 0
            res = None
            for i in range(pos, len(s)):
                r1 = recur(i + 1, remaining - 1)
                if r1 is None: break
                d = int(s[i]) * 10 ** (remaining - 1)
                res = max(res or 0, d + r1)
            return res
        return recur(0, 12)
    return sum(solve(s) for s in data)


def problem4(data, second):
    # if second: return
    _data = split_data('''
..@@.@@@@.
@@@.@.@.@@
@@@@@.@.@@
@.@@@@..@.
@@.@@@@.@@
.@@@@@@@.@
.@.@.@.@@@
@.@@@.@@@@
.@@@@@@@@.
@.@.@@@.@.
    ''')

    # field = ndarray_from_chargrid(data)
    field = np.array([list(row) for row in data], dtype='U1')
    field = (field == '@').astype('int8')
    field = np.pad(field, 1)

    if not second:
        nb = sum(np.roll(field, (i - 1, j - 1), axis=(0, 1))
                 for i in range(3) for j in range(3) if i != 1 or j != 1)
        field = field & (nb < 4)
        return np.sum(field)

    removed = 0
    while True:
        nb = sum(np.roll(field, (i - 1, j - 1), axis=(0, 1))
                 for i in range(3) for j in range(3) if i != 1 or j != 1)
        removable = field & (nb < 4)
        cnt = np.sum(removable)
        if not cnt:
            return removed
        removed += cnt
        field ^= removable


def problem5(data, second):
    # if second: return
    _data = split_data('''
3-5
10-14
16-20
12-18

1
5
8
11
17
32
    ''')
    ranges = []
    ingrs = []
    for s in data:
        if not s:
            continue
        ss = s.split('-')
        if len(ss) == 2:
            ranges.append(range(int(ss[0]), int(ss[1]) + 1))
        else:
            assert len(ss) == 1
            ingrs.append(int(ss[0]))
    if not second:
        fresh = 0
        for i in ingrs:
            for r in ranges:
                if i in r:
                    fresh += 1
                    break
        return fresh

    def union(r1, r2):
        if r1.start >= r2.start and r1.stop <= r2.stop:
            return r2, True
        if r2.start >= r1.start and r2.stop <= r1.stop:
            return r1, True
        if r1.start >= r2.start and r1.stop >= r2.stop and r2.stop >= r1.start:
            return range(r2.start, r1.stop), True
        if r2.start >= r1.start and r2.stop >= r1.stop and r1.stop >= r2.start:
            return range(r1.start, r2.stop), True
        return r2, False

    disjoint = []
    for r in ranges:
        d2 = []
        for d in disjoint:
            r2, joined = union(d, r)
            if joined:
                r = r2
            else:
                d2.append(d)
        d2.append(r)
        disjoint = d2

    return sum(r.stop - r.start for r in disjoint)


def problem6(data, second):
    # if second: return
    data = '''
123 328  51 64
 45 64  387 23
  6 98  215 314
*   +   *   +
'''
    data = get_raw_data()
    data = [s for s in data.split('\n') if s.strip()]
    if not second:
        data = [s.split() for s in data]
        assert all(len(d) == len(data[0]) for d in data)
        res = 0
        for i, op in enumerate(data[-1]):
            op = {'+' : operator.add, '*': operator.mul}[op]
            ds = [int(data[j][i]) for j in range(len(data) - 1)]
            res += functools.reduce(op, ds)
        return res
    # second
    l = max(len(s) for s in data)
    data = [s + ' ' * (l - len(s)) for s in data]
    res = 0
    cols_src = list(re.finditer('[+*]\\s*', data[-1]))
    for col, m in enumerate(cols_src):
        # print(col, m)
        ds = []
        for digit_pos in range(m.span()[0], m.span()[1]):
            s = ''
            for i in range(len(data) - 1):
                s += data[i][digit_pos]
            s = s.strip()
            if s: ds.append(int(s))
        op = m.group()[0]
        op = {'+' : operator.add, '*': operator.mul}[op]
        res += functools.reduce(op, ds)
    return res




def problem7(data, second):
    # if second: return
    _data = split_data('''.......S.......
...............
.......^.......
...............
......^.^......
...............
.....^.^.^.....
...............
....^.^...^....
...............
...^.^...^.^...
...............
..^...^.....^..
...............
.^.^.^.^.^...^.
...............''')
    beampos = {data[0].find('S'): 1}
    assert beampos
    cnt = 0
    for line in data:
        if '^' in line:
            newb = Counter()
            for b, c in beampos.items():
                if line[b] == '^':
                    cnt += 1
                    newb[b - 1] += c
                    newb[b + 1] += c
                else:
                    newb[b] += c
            beampos = newb
    if second:
        return sum(beampos.values())
    return cnt





##########

def problem(data, second):
    if second: return
    data = split_data(''' ''')


##########

if __name__ == '__main__':
    print('Hello')
    solve_latest()
    # solve_latest(6)
    # solve_all()
