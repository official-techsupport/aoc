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
        if not s: continue
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


def problem7(data, second):
    _data = split_data('''32T3K 765
T55J5 684
KK677 28
KTJJT 220
QQQJA 483''')

    def hand_type(hand):
        if second and any(hand):
            [[top, _]] = Counter(filter(None, hand)).most_common(1)
            hand = [top if not c else c for c in hand]

        d = Counter(hand)
        if len(d) == 1:
            return 7
        (_, first), (_, sec) = d.most_common(2)
        if first == 4:
            return 6
        if first == 3 and sec == 2:
            return 5
        if first == 3:
            return 4
        if first == 2 and sec == 2:
            return 3
        if first == 2:
            return 2
        return 1

    hands = []
    for s in data:
        hand, bid = s.split()
        hand = [('J23456789TQKA' if second else '23456789TJQKA').index(c) for c in hand]
        t = hand_type(hand)
        hands.append((t, hand, int(bid)))
    hands.sort()
    res = 0
    for rank0, (t, hand, bid) in enumerate(hands):
        res += bid * (rank0 + 1)
    return res


def problem8(data, second):
    _data = split_data('''LR

11A = (11B, XXX)
11B = (XXX, 11Z)
11Z = (11B, XXX)
22A = (22B, XXX)
22B = (22C, 22C)
22C = (22Z, 22Z)
22Z = (22B, 22B)
XXX = (XXX, XXX)''')

    moves = data[0]
    nodes = {}
    for s in data[1:]:
        if not s: continue
        src, dst = s.split('=')
        dst = dst.strip('() ')
        dst1, dst2 = dst.split(',')
        nodes[src.strip()] = (dst1.strip(), dst2.strip())

    if not second:
        cur = 'AAA'
        for steps, c in enumerate(itertools.cycle(moves)):
            cur = nodes[cur][0 if c == 'L' else 1]
            if cur == 'ZZZ':
                break
        return steps + 1

    arr = []
    for start in nodes.keys():
        if start.endswith('A'):
            cur = start
            for steps, c in enumerate(itertools.cycle(moves)):
                if cur.endswith('Z'):
                    break
                cur = nodes[cur][0 if c == 'L' else 1]
            arr.append(steps)

    return math.lcm(*arr)


def problem9(data, second):
    _data = split_data('''
0 3 6 9 12 15
1 3 6 10 15 21
10 13 16 21 30 45
''')

    def derive(lst):
        return [b - a for a, b in pairwise(lst)]

    def extrapolate(lst):
        res = [lst]
        while any(lst):
            lst = derive(lst)
            res.append(lst)
        return res

    res = 0
    for s in data:
        lst = list(map(int, s.split()))
        stack = extrapolate(lst)

        if second:
            for i in reversed(range(len(stack) - 1)):
                stack[i].append(stack[i][0] - stack[i + 1][-1])
        else:
            for i in reversed(range(len(stack) - 1)):
                stack[i].append(stack[i][-1] + stack[i + 1][-1])
        res += stack[0][-1]

    return res


def problem10(data, second):
    return
    data = split_data('''
.....
.S-7.
.|.|.
.L-J.
.....
''')
    if second: return
    g = networkx.Graph()

    def add_edge(row, col, dir):
        rc = row, col
        if dir == 0:
            g.add_edge(rc, (row - 1, col))
        if dir == 1:
            g.add_edge(rc, (row, col + 1))
        if dir == 2:
            g.add_edge(rc, (row + 1, col))
        if dir == 3:
            g.add_edge(rc, (row, col - 1))


    start = None

    for row, s in enumerate(data):
        for col, c in enumerate(s):
            if c == '|':
                add_edge(row, col, 0)
                add_edge(row, col, 2)
            if c == '-':
                add_edge(row, col, 1)
                add_edge(row, col, 3)
            if c == 'L':
                add_edge(row, col, 0)
                add_edge(row, col, 1)
            if c == 'J':
                add_edge(row, col, 0)
                add_edge(row, col, 3)
            if c == '7':
                add_edge(row, col, 2)
                add_edge(row, col, 3)
            if c == 'F':
                add_edge(row, col, 2)
                add_edge(row, col, 1)
            if c == "S":
                start = (row, col)


    for dir1 in range(4):
        for dir2 in range(4):
            if dir2 <= dir1: continue
            oldg = g.copy()

    print(g, 'hey hey hey')
    return None


def problem11(data, second):
    _data = split_data('''
...#......
.......#..
#.........
..........
......#...
.#........
.........#
..........
.......#..
#...#.....''')
    g = ndarray_from_chargrid(data)
    empty_rows = []
    for i, s in enumerate(g):
        if all(c == '.' for c in s):
            empty_rows.append(i)
    empty_cols = []
    for i, s in enumerate(g.transpose()):
        if all(c == '.' for c in s):
            empty_cols.append(i)
    galaxies = np.argwhere(g == '#')

    res = 0
    for a in galaxies:
        for b in galaxies:
            extra = sum(a[0] <= r <= b[0] for r in empty_rows)
            extra += sum(a[1] <= r <= b[1] for r in empty_cols)
            res += abs(a[0] - b[0]) + abs(a[1] - b[1]) + extra * ((1000000 * 2 - 2) if second else 2)
            # the `extra` calculation above is broken half of the time.

    return res // 2


def problem13(data, second):
    _data = split_data('''#.##..##.
..#.##.#.
##......#
##......#
..#.##.#.
..##..##.
#.#.##.#.

#...##..#
#....#..#
..##..###
#####.##.
#####.##.
..##..###
#....#..#''')

    def is_mirrored(m):
        def check(i):
            cnt = min(i, len(m) - i)
            assert cnt
            s = 0
            for k in range(cnt):
                s += np.sum(m[i - k - 1] != m[i + k])
            if s == (1 if second else 0):
                return i
        for i in range(1, len(m)):
            c = check(i)
            if c: return c

    res = []
    cnt = 0
    def process(m):
        # nonlocal cnt
        # print(cnt)
        # print('\n'.join(m))
        # cnt += 1

        m = ndarray_from_chargrid(m)
        c = is_mirrored(m.transpose())
        if c:
            res.append(c)
            return
        c = is_mirrored(m)
        if c:
            res.append(c * 100)
            return
        assert False

    cur = []
    for s in data:
        if not s:
            process(cur)
            cur = []
        else:
            cur.append(s)
    process(cur)
    print(res)
    return sum(res)


def problem14(data, second):
    _data = split_data('''O....#....
O.OO#....#
.....##...
OO.#O....O
.O.....O#.
O.#..O.#.#
..O..#O..O
.......O..
#....###..
#OO..#....''')
    data = ndarray_from_chargrid(data)

    def shift():
        changed = frozenset(range(len(data)))
        while changed:
            newchanged = set()
            for i, s in enumerate(data[1:]):
                i += 1
                if i not in changed: continue
                for x, c in enumerate(s):
                    if c == 'O' and data[i - 1][x] == '.':
                        data[i - 1][x] = 'O'
                        data[i][x] = '.'
                        newchanged.add(i - 1)
            changed = newchanged

    def load():
        res = 0
        h = len(data)
        for i, s in enumerate(data):
            for c in s:
                if c == 'O':
                    res += h
            h -= 1
        return res

    if not second:
        shift()
        return load()

    hashes = {}

    def cycle():
        nonlocal data
        for _ in range(4):
            shift()
            data = np.rot90(data, -1)
        return hash(data.tostring())

    target_cycles = 1000000000
    for c in range(target_cycles):
        h = cycle()
        if h in hashes:
            break
        hashes[h] = c
        print(c, len(hashes))

    cycle_len = c - hashes[h]
    print(cycle_len)
    target_cycles -= c + 1
    target_cycles %= cycle_len
    for c in range(target_cycles):
        cycle()

    return load()


def problem15(data, second):
    _data = split_data('''rn=1,cm-,qp=3,cm=2,qp-,pc=4,ot=9,ab=5,pc-,pc=6,ot=7''')

    def holiday_ash(s):
        r = 0
        for c in s:
            r += ord(c)
            r *= 17
            r %= 256
        return r
    data = data.split(',')
    if not second:
        return sum(holiday_ash(s) for s in data)

    boxes = [[] for _ in range(256)]
    for s in data:
        if s.endswith('-'):
            label = s[:-1]
            h = holiday_ash(label)
            box = boxes[h]
            for i, (c, _) in enumerate(box):
                if c == label:
                    del box[i]
                    break
        else:
            label, f = s.split('=')
            h = holiday_ash(label)
            box = boxes[h]
            for i, (c, _) in enumerate(box):
                if c == label:
                    box[i] = (label, f)
                    break
            else:
                box.append((label, f))

    res = 0
    for i, box in enumerate(boxes):
        for j, (_, f) in enumerate(box):
            res += (i + 1) * (j + 1) * int(f)
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
    # solve_latest(7)
