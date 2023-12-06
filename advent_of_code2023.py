#!/usr/bin/env python3

import advent_of_code_utils
from advent_of_code_utils import *
utils_init(2023, globals())

###########


def problem1(data, second):
    if second: _data = split_data('''two1nine
eightwothree
abcone2threexyz
xtwone3four
4nineeightseven2
zoneight234
7pqrstsixteen''')
    ds = '\\d one two three four five six seven eight nine'.split()
    if not second:
        del ds[1:]
    rx1 = re.compile('.*?(' + '|'.join(ds) + ')')
    rx2 = re.compile('.*(' + '|'.join(ds) + ')')
    def get(s: str):
        m1 = rx1.match(s).group(1)
        m2 = rx2.match(s).group(1)
        def decode(s):
            try:
                return str(ds.index(s))
            except ValueError:
                return str(int(s))
        return int(decode(m1) + decode(m2))
    return sum(get(s) for s in data)


def problem2(data, second):
    _data = split_data('''Game 1: 3 blue, 4 red; 1 red, 2 green, 6 blue; 2 green
Game 2: 1 blue, 2 green; 3 green, 4 blue, 1 red; 1 green, 1 blue
Game 3: 8 green, 6 blue, 20 red; 5 blue, 4 red, 13 green; 5 green, 1 red
Game 4: 1 green, 3 red, 6 blue; 3 green, 6 red; 3 green, 15 blue, 14 red
Game 5: 6 red, 1 blue, 3 green; 2 blue, 1 red, 2 green''')
    d = []
    for i, s in enumerate(data):
        g, r = s.split(':')
        g = int(g.removeprefix('Game '))
        assert g == i + 1
        r = r.split(';')
        measures = []
        for s in r:
            st = s.split(',')
            m = {}
            for it in st:
                n, color = it.split()
                m[color] = int(n)
            measures.append(m)
        d.append(measures)
    target = {'red': 12, 'green': 13, 'blue': 14}
    res = 0
    for id, mm in enumerate(d):
        def check(mm):
            for m in mm:
                for color, n in m.items():
                    if n > target[color]:
                        return 0
            return id + 1
        def check2(mm):
            mins = {}
            for m in mm:
                for color, n in m.items():
                    mins[color] = max(mins[color], n) if color in mins else n
            # print(id, mins)
            r = 1
            for it in mins.values():
                r *= it
            return r
        if second:
            res += check2(mm)
        else:
            res += check(mm)
    return res


def problem3(data, second):
    _data = split_data('''
467..114..
...*......
..35..633.
......#...
617*......
.....+.58.
..592.....
......755.
...$.*....
.664.598..
''')

    rows, cols = len(data), len(data[0])
    visited = set()
    digits = string.digits
    numbers = []
    gears = defaultdict(list)

    def get(i, j):
        if 0 <= i < rows and 0 <= j < cols:
            return data[i][j]
        return '.'

    for i in range(rows):
        for j in range(cols):
            if (i, j) in visited: continue
            if data[i][j] in digits:
                visited.add((i, j))
                n = int(data[i][j])
                for k in range(1, 10):
                    if get(i, j + k) in digits:
                        visited.add((i, j + k))
                        n = n * 10 + int(data[i][j + k])
                    else:
                        break
                found = False
                current = set()
                for k in range(k):
                    for dx in (-1, 0, 1):
                        for dy in (-1, 0, 1):
                            c = get(i + dx, j + k + dy)
                            if c is not None and c not in digits and c != '.':
                                found = True
                                if c == '*':
                                    coord = (i + dx, j + k + dy)
                                    if coord not in current:
                                        gears[coord].append(n)
                                        current.add(coord)

                if found:
                    numbers.append(n)

    if not second:
        return sum(numbers)

    res = 0
    for lst in gears.values():
        if len(lst) == 2:
            res += lst[0] * lst[1]
    return res


def problem4(data, second):
    _data = split_data('''Card 1: 41 48 83 86 17 | 83 86  6 31 17  9 48 53
Card 2: 13 32 20 16 61 | 61 30 68 82 17 32 24 19
Card 3:  1 21 53 59 44 | 69 82 63 72 16 21 14  1
Card 4: 41 92 73 84 69 | 59 84 76 51 58  5 54 83
Card 5: 87 83 26 28 32 | 88 30 70 12 93 22 82 36
Card 6: 31 18 13 56 72 | 74 77 10 23 35 67 36 11''')
    res = []
    for s in data:
        _, s = s.split(':')
        winning, have = s.split('|')
        def parse(s):
            return set(map(int, s.split()))
        winning, have = parse(winning), parse(have)
        res.append(have & winning)
    if not second:
        return sum(0 if not r else 2**(len(r) - 1) for r in res)

    copies = [1] * len(res)
    for i, r in enumerate(res):
        for x in range(i + 1, i + len(r) + 1):
            copies[x] += copies[i]
    return sum(copies)


def problem5(data, second):
    _data = split_data('''seeds: 79 14 55 13

seed-to-soil map:
50 98 2
52 50 48

soil-to-fertilizer map:
0 15 37
37 52 2
39 0 15

fertilizer-to-water map:
49 53 8
0 11 42
42 0 7
57 7 4

water-to-light map:
88 18 7
18 25 70

light-to-temperature map:
45 77 23
81 45 19
68 64 13

temperature-to-humidity map:
0 69 1
1 0 69

humidity-to-location map:
60 56 37
56 93 4''')

    assert data[0].startswith('seeds:')
    seeds = tuple(int(x) for x in data[0].split(':')[1].split())
    del data[0]
    maps = []
    curmap = None
    for s in data:
        if 'map:' in s:
            curmap = []
            maps.append(curmap)
        else:
            vals = tuple(int(x) for x in s.split())
            assert len(vals) == 3
            curmap.append(vals)

    def mapstep(x, curmap):
        for dest, src, rlen in curmap:
            if src <= x < src + rlen:
                return dest + x - src
        return x

    def intersect_range(x, xlen, y, ylen):
        r = max(x, y)
        rend = min(x + xlen, y + ylen)
        if rend <= r:
            return None, None
        return r, rend - r

    def mapstep2(x, xlen, curmap):
        ranges = []
        srcranges = []
        for dest, src, rlen in curmap:
            r, rlen = intersect_range(x, xlen, src, rlen)
            if r is None: continue
            srcranges.append((r, rlen))
            ranges.append((dest + r - src, rlen))

        curx = x
        for r, rlen in sorted(srcranges):
            if r - curx > 0:
                ranges.append((curx, r - curx))
            curx = r + rlen
        if curx < x + xlen:
            ranges.append((curx, x + xlen - curx))

        return ranges

    if not second:
        for curmap in maps:
            seeds = [mapstep(x, curmap) for x in seeds]
        return min(seeds)

    seeds = list(grouper(seeds, 2))
    for curmap in maps:
        seeds = [y for x in seeds for y in mapstep2(*x, curmap)]
        # print(seeds)
    return min(x for (x, _) in seeds)


def problem6(data, second):
    _data = split_data('''Time:      7  15   30
Distance:  9  40  200''')
    def parse(prefix, s):
        raw = removeprefix(prefix, s).split()
        if second:
            return [int(''.join(raw))]
        return [int(s) for s in raw]
    times = parse('Time:', data[0])
    distances = parse('Distance:', data[1])

    def ways(time, distance):
        res = 0
        for charge in range(0, time + 1):
            dst = charge * (time - charge)
            if dst > distance:
                # print(time, distance, charge)
                res += 1
        # print(time, distance, res)
        return res

    return functools.reduce(operator.mul, (ways(t, d) for t, d in zip(times, distances)))


#########

def problem(data, second):
    data = split_data(''' ''')
    if second: return
    return None


##########

if __name__ == '__main__':
    print('Hello')
    # solve_all()
    # solve_latest(15)
    solve_latest()
