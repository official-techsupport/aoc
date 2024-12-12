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


def problem6(data, second):
    data = split_data('''
....#.....
.........#
..........
..#.......
.......#..
..........
.#..^.....
........#.
#.........
......#...''')
    g = ndarray_from_chargrid(data)
    height, width = len(g), len(g[0])
    [pos] = np.argwhere(g == '^')
    pos = pos[1] + 1j*pos[0]
    d = -1j
    obstructions = 0
    def step(pos, d):
        nonlocal obstructions
        pv2 = c2v2(pos)
        cross = g[pv2] == 'X'
        g[pv2] = "X"
        while True:
            pos2 = pos + d
            if not (pos2.real in range(width) and pos2.imag in range(height)):
                return None, None
            if g[c2v2(pos2)] == '#':
                d *= 1j
                continue
            if cross:
                print(pos)
                obstructions += 1
            break
        return pos2, d
    while pos is not None:
        pos, d = step(pos, d)
    # print(g)
    if second:
        return obstructions
    return np.sum(g == 'X')


def problem7(data, second):
    # if second: return
    _data = split_data('''190: 10 19
3267: 81 40 27
83: 17 5
156: 15 6
7290: 6 8 6 15
161011: 16 10 13
192: 17 8 14
21037: 9 7 18 13
292: 11 6 16 20''')
    results = []
    nums = []
    for s in data:
        r, s = s.split(':')
        s = s.split()
        results.append(int(r))
        nums.append([int(it) for it in s])

    def can_get(r, ns):
        ns = iter(ns)
        dyn = {next(ns)}
        for n in ns:
            newdyn = set()
            for i in dyn:
                x = i + n
                if x <= r: newdyn.add(x)
                x = i * n
                if x <= r: newdyn.add(x)
                if second:
                    x = int(str(i) + str(n))
                    if x <= r: newdyn.add(x)
            dyn = newdyn
        if r in dyn:
            return r
        return False

    return sum(can_get(r, ns) for r, ns in zip(results, nums))


def problem8(data, second):
    # if second: return
    _data = split_data('''............
........0...
.....0......
.......0....
....0.......
......A.....
............
............
........A...
.........A..
............
............''')

    height, width = len(data), len(data[0])

    nodes = defaultdict(list)
    for row, line in enumerate(data):
        for col, c in enumerate(line):
            if c != '.':
                nodes[c].append(row + 1j*col)

    def antinodes(n1, n2):
        d = n2 - n1
        return [n2 + d, n1 - d]

    def antinodes2(n1, n2):
        d = n2 - n1
        def go(p, d):
            while 0 <= p.real < height and 0 <= p.imag < width:
                yield p
                p += d
        yield from go(n2, d)
        yield from go(n1, -d)

    res = set()
    f = antinodes2 if second else antinodes
    for nn in nodes.values():
        for n1, n2 in itertools.combinations(nn, 2):
            res.update(f(n1, n2))

    res = [p for p in res if 0 <= p.real < height and 0 <= p.imag < width]
    return len(res)


def problem9(data, second):
    _data = split_data('''2333133121414131402''')
    assert len(data) % 2
    dd = np.zeros(len(data) + 1, dtype=np.int8)
    for i, c in enumerate(data):
        dd[i] = int(c)
    size = sum(dd)
    disk = -1 * np.ones(size, dtype=np.int32)
    idx = 0
    freelist = []
    filelist = []
    for blockid, (b, e) in enumerate(itertools.batched(dd, 2)):
        if b:
            filelist.append((blockid, idx, b))
            disk[idx:idx + b] = blockid
            idx += b
        if e:
            freelist.append((idx, e))
            idx += e

    if not second:
        idx1 = 0
        idx2 = len(disk) - 1
        while True:
            while disk[idx2] < 0:
                idx2 -= 1
            while disk[idx1] >= 0:
                idx1 += 1
            if idx1 > idx2:
                break
            disk[idx1] = disk[idx2]
            disk[idx2] = -1
    else:
        for blockid, idx1, l1 in reversed(filelist):
            for i, (idx2, l2) in enumerate(freelist):
                if idx2 > idx1:
                    break
                if l2 >= l1:
                    disk[idx2 : idx2 + l1] = blockid
                    disk[idx1 : idx1 + l1] = -1
                    if l2 == l1:
                        del freelist[i]
                    else:
                        freelist[i] = (idx2 + l1, l2 - l1)
                    break

    return sum(i * int(blockid) for i, blockid in enumerate(disk) if blockid > 0)


def problem10(data, second):
    if not second: return
    _data = split_data('''
89010123
78121874
87430965
96549874
45678903
32019012
01329801
10456732''')
    g = networkx.DiGraph()
    height, width = len(data), len(data[0])
    starts, ends = [], []
    for r in range(height):
        for c in range(width):
            p1 = (r, c)
            d1 = data[r][c]
            if d1 == '0':
                starts.append(p1)
            if d1 == '9':
                ends.append(p1)
            for d in directions4:
                p2 = addv2(p1, d)
                if not (p2[0] in range(height) and p2[1] in range(width)):
                    continue
                d2 = data[p2[0]][p2[1]]
                if d1 == '.' or d2 == '.':
                    continue
                if int(d2) - int(d1) == 1:
                    g.add_edge(p1, p2)
    sp = dict(networkx.all_pairs_all_shortest_paths(g))
    if second:
        return sum(sum(len(sp[s][e]) for e in ends if e in sp[s]) for s in starts)
    return sum(sum(1 for e in ends if e in sp[s]) for s in starts)


def problem11(data, second):
    _data = split_data('''125 17''')
    stones = [int(s) for s in data.split()]
    def blink1(s):
        if s == 0:
            return [1]
        st = str(s)
        if not len(st) & 1:
            return [
                int(st[0 : len(st) // 2]),
                int(st[len(st) // 2 : ])]
        return [s * 2024]

    def blink2(ss):
        r = defaultdict(int)
        for s, cnt in ss.items():
            for s2 in blink1(s):
                r[s2] += cnt
        return r

    stones = Counter(stones)
    for _ in range(75 if second else 25):
        stones = blink2(stones)

    return sum(stones.values())


def problem12(data, second):
    _data = split_data('''
RRRRIICCFF
RRRRIICCCF
VVRRRCCFFF
VVRCCCJFFF
VVVVCJJCFE
VVIVCCJJEE
VVIIICJJEE
MIIIIIJJEE
MIIISIJEEE
MMMISSJEEE''')
    _data = split_data('''
AAAAAA
AAABBA
AAABBA
ABBAAA
ABBAAA
AAAAAA
''')

    height = len(data)
    width = len(data[0])

    def inside(p):
        return p[0] in range(height) and p[1] in range(width)

    def get_region(p):
        letter = data[p[0]][p[1]]
        visited = set([p])
        front = deque(visited)
        while front:
            p = front.pop()
            for d in directions4:
                p2 = addv2(p, d)
                if inside(p2) and not p2 in visited and data[p2[0]][p2[1]] == letter:
                    front.append(p2)
                    visited.add(p2)
        return visited

    def perimeter(region: set):
        p = next(iter(region))
        letter = data[p[0]][p[1]]
        return sum(not inside(p2) or data[p2[0]][p2[1]] != letter for p2 in
            (addv2(p, d) for p in region for d in directions4))

    def sides(region: set):
        p = next(iter(region))
        letter = data[p[0]][p[1]]
        ss = set()
        for p in region:
            for d in ((0, -1), (1, 0), (0, 1), (-1, 0)):
                p2 = addv2(p, d)
                if inside(p2) and data[p2[0]][p2[1]] == letter:
                    continue
                ss.add((p, d))
        return sum((addv2(p, (1, 0)), d) not in ss and
                   (addv2(p, (0, 1)), d) not in ss
                   for p, d in ss)

    remaining = {(r, c) for r in range(height) for c in range(width)}
    rs = []
    while remaining:
        region = get_region(remaining.pop())
        remaining -= region
        rs.append(region)
    if second:
        return sum(len(r) * sides(r) for r in rs)
    return sum(len(r) * perimeter(r) for r in rs)



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
