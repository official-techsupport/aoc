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
    _data = split_data('''
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
    pos = pos[0] + 1j*pos[1]
    d = -1
    visited = set()
    LOOP = object()
    obsset = set()
    def inside(pos):
        return pos.real in range(width) and pos.imag in range(height)
    def step(pos, d):
        pv2 = c2v2(pos)
        g[pv2] = "X"
        p = (pos, d)
        if p in visited:
            return LOOP, None
        visited.add(p)
        pos2 = pos + d
        if not inside(pos2):
            return None, None
        if g[c2v2(pos2)] == '#':
            d *= -1j
        else:
            pos = pos2
        return pos, d

    def run():
        nonlocal pos, d
        while pos is not None and pos != LOOP:
            pos, d = step(pos, d)
        return pos

    if not second:
        run()
        return np.sum(g == 'X')

    obstructions = 0
    while True:
        pos, d = step(pos, d)
        if pos is None:
            break
        pos2 = pos + d
        if inside(pos2) and g[c2v2(pos2)] == '.' and pos2 not in obsset:
            visited = set()
            shadowpos, shadowd, shadowg = pos, d, np.copy(g)
            g[c2v2(pos2)] = '#'
            res = run()
            if res == LOOP:
                obstructions += 1
                print(obstructions)
                obsset.add(pos2)
            pos, d, g = shadowpos, shadowd, shadowg
            visited = set()
    return obstructions




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
    dirs = (1, 1j, -1, -1j)

    def inside(p):
        return p.real in range(height) and p.imag in range(width)

    def get(p):
        return data[int(p.real)][int(p.imag)]

    def get_region(p):
        letter = get(p)
        visited = set([p])
        front = deque(visited)
        while front:
            p = front.pop()
            for d in dirs:
                p2 = p + d
                if inside(p2) and not p2 in visited and get(p2) == letter:
                    front.append(p2)
                    visited.add(p2)
        return visited

    def perimeter(region: set):
        p = next(iter(region))
        letter = get(p)
        return sum(not inside(p2) or get(p2) != letter for p2 in
            (p + d for p in region for d in dirs))

    def sides(region: set):
        p = next(iter(region))
        letter = get(p)
        ss = set()
        for p in region:
            for d in dirs:
                p2 = p + d
                if not inside(p2) or get(p2) != letter:
                    ss.add((p, d))
        return sum((p + 1, d) not in ss and
                   (p + 1j, d) not in ss
                   for p, d in ss)

    remaining = {r + 1j*c for r in range(height) for c in range(width)}
    rs = []
    while remaining:
        region = get_region(remaining.pop())
        remaining -= region
        rs.append(region)
    if second:
        return sum(len(r) * sides(r) for r in rs)
    return sum(len(r) * perimeter(r) for r in rs)


def problem13(data, second):
    _data = split_data('''
Button A: X+94, Y+34
Button B: X+22, Y+67
Prize: X=8400, Y=5400

Button A: X+26, Y+66
Button B: X+67, Y+21
Prize: X=12748, Y=12176

Button A: X+17, Y+86
Button B: X+84, Y+37
Prize: X=7870, Y=6450

Button A: X+69, Y+23
Button B: X+27, Y+71
Prize: X=18641, Y=10279
''')
    data = iter(data)
    def split1(s, sep):
        _, s = s.split(':')
        x, y = s.split(',')
        x = int(x.split(sep)[1])
        y = int(y.split(sep)[1])
        return x, y
    machines = []
    for s in data:
        if not(s):
            s = next(data)
        x1, y1 = split1(s, '+')
        s = next(data)
        x2, y2 = split1(s, '+')
        s = next(data)
        x3, y3 = split1(s, '=')
        machines.append((x1, y1, x2, y2, x3, y3))
    import z3
    res = 0
    addend = 10000000000000 if second else 0
    a, b = z3.Ints('a b')
    zs = z3.Solver()
    zs.add(a >= 0)
    zs.add(b >= 0)
    for x1, y1, x2, y2, x3, y3 in machines:
        zs.push()
        zs.add(x1 * a + x2 * b == x3 + addend)
        zs.add(y1 * a + y2 * b == y3 + addend)
        if zs.check() == z3.sat:
            res += zs.model().eval(a * 3 + b).as_long()
        zs.pop()
    return res


def problem14(data, second):
    # if second: return
    _data = split_data('''p=0,4 v=3,-3
p=6,3 v=-1,-3
p=10,3 v=-1,2
p=2,0 v=2,-1
p=0,0 v=1,3
p=3,0 v=-2,-2
p=7,6 v=-1,-3
p=3,0 v=-1,-2
p=9,3 v=2,3
p=7,3 v=-1,2
p=2,4 v=2,-3
p=9,5 v=-3,-3''')
    width, height = 11, 7
    width, height = 101, 103
    robots = []
    w2 = width // 2
    h2 = height // 2
    steps = 100
    def split2(s):
        _, s = s.split('=')
        a, b = s.split(',')
        return int(a), int(b)
    for s in data:
        a, b = s.split()
        robots.append(split2(a) + split2(b))
    quadrants = defaultdict(int)
    def quadrant(x, y):
        if x == w2 or y == h2:
            return 4
        return (x < w2) * 2 + (y < h2)

    if not second:
        for x, y, dx, dy in robots:
            x2 = (x + dx * steps) % width
            y2 = (y + dy * steps) % height
            quadrants[quadrant(x2, y2)] += 1
        return quadrants[0] * quadrants[1] * quadrants[2] * quadrants[3]

    import zlib
    most = 1000000
    for steps in range(1, 10000):
        quadrants = defaultdict(int)
        d = np.zeros((height, width))
        coords = set()
        for x, y, dx, dy in robots:
            x2 = (x + dx * steps) % width
            y2 = (y + dy * steps) % height
            d[y2, x2] = 1
            coords.add((y2, x2))

        # l = len(zlib.compress(d, level=1))
        # if l < most:
        #     most = l
        #     print('\n'.join(''.join('#' if d[row][col] else '.' for col in range(width)) for row in range(height)))
        #     print(steps, most)

        if len(coords) == len(robots):
            return steps


def problem15(data, second):
    # if second: return
    _data = split_data('''
#######
#...#.#
#.....#
#..OO@#
#..O..#
#.....#
#######

<vv<<^^<<^^''')

    _data = split_data('''
##########
#..O..O.O#
#......O.#
#.OO..O.O#
#..O@..O.#
#O#..O...#
#O..O..O.#
#.OO.O.OO#
#....O...#
##########

<vv>^<v^>v>^vv^v>v<>v^v<v<^vv<<<^><<><>>v<vvv<>^v^>^<<<><<v<<<v^vv^v>^
vvv<<^>^v^^><<>>><>^<<><^vv^^<>vvv<>><^^v>^>vv<>v<<<<v<^v>^<^^>>>^<v<v
><>vv>v^v^<>><>>>><^^>vv>v<^^^>>v^v^<^^>v^^>v^<^v>v<>>v^v^<v>v^^<^^vv<
<<v<^>>^^^^>>>v^<>vvv^><v<<<>^^^vv^<vvv>^>v<^^^^v<>^>vvvv><>>v^<<^^^^^
^><^><>>><>^^<<^^v>>><^<v>^<vv>>v>>>^v><>^v><<<<v>>v<v<v>vvv>^<><<>^><
^>><>^v<><^vvv<^^<><v<<<<<><^v<<<><<<^^<v<^^^><^>>^<v^><<<^>>^v<v^v<v^
>^>>^v>vv>^<<^v<>><<><<v<<v><>v<^vv<<<>^^v^>^^>>><<^v>>v^v><^^>>^<>vv^
<><^^>^^^<><vvvvv^v<v<<>^v<v>v<<^><<><<><<<^^<<<^<<>><<><^^^>^^<>^>v<>
^^>vv<^v^v<vv>^<><v<^v>^^^>>>^^vvv^>vvv<>>>^<^>>>>>^<<^v>^vvv<>^<><<v>
v^^>>><<^^<>>^v^<v^vv<>v^<<>^<^v^v><^<<<><<^<v><v<>vv>>v><v^<vv<>v^<<^''')
    data = iter(data)
    g = []
    while line := next(data).strip():
        if second:
            g.append(''.join({'.': '..', '#': '##', 'O': '[]', '@': '@.'}[c] for c in line))
        else:
            g.append(line)
    moves = ''.join(s for s in data)
    g = np_cidx(ndarray_from_chargrid(g))
    mdir = {'<': -1j, '>': 1j, '^': -1, 'v': 1}
    [pos] = np.argwhere(g == '@')
    pos = pos[0] + 1j * pos[1]
    g[pos] = '.'
    height, width = len(g), len(g[0])
    def inside(pos):
        return pos.real in range(width) and pos.imag in range(height)

    def move(pos, d):
        pos2 = pos + d
        if (c := g[pos2]) == '#':
            return pos
        if c == '.':
            return pos2
        assert c == 'O'
        pos3 = pos2
        while True:
            pos3 += d
            c = g[pos3]
            if c == '#':
                return pos
            if c == '.':
                g[pos2] = '.'
                g[pos3] = 'O'
                return pos2

    def move2(pos, d):
        front = [pos]
        fronts = []
        while True:
            front = [p + d for p in front]
            if any(g[p] == '#' for p in front):
                return pos
            if all(g[p] == '.' for p in front):
                for front in reversed(fronts):
                    for p in front:
                        g[p + d] = g[p]
                        g[p] = '.'
                return pos + d

            if d.real:
                front = set().union(*(
                    [p, p + 1j if c == '[' else p - 1j]
                    for p in front
                    if (c := g[p]) in '[]'))
            fronts.append(front)

    # print(g)
    mv = move2 if second else move
    for m in moves:
        pos = mv(pos, mdir[m])
        # g[pos] = '@'
        # print(g)
        # g[pos] = '.'

    c = '[' if second else 'O'
    return sum(100*p[0] + p[1] for p in np.argwhere(g == c))


def problem16(data, second):
    # if second: return
    _data = split_data('''
#################
#...#...#...#..E#
#.#.#.#.#.#.#.#.#
#.#.#.#...#...#.#
#.#.#.#.###.#.#.#
#...#.#.#.....#.#
#.#.#.#.#.#####.#
#.#...#.#.#.....#
#.#.#####.#.###.#
#.#.#.......#...#
#.#.###.#####.###
#.#.#...#.....#.#
#.#.#.#####.###.#
#.#.#.........#.#
#.#.#.#########.#
#S#.............#
#################''')
    g = np_cidx(ndarray_from_chargrid(data))
    [startpos] = np.argwhere(g == 'S')
    startpos = v22c(startpos)
    [endpos] = np.argwhere(g == 'E')
    endpos = v22c(endpos)
    g[startpos] = g[endpos] = '.'

    class posd(namedtuple('posd', 'pos d')):
        def __lt__(self, other):
            return False

    pd = posd(startpos, 1j)
    front = [(0, pd, None)]
    visited = {}
    paths = defaultdict(list)

    def unwind(pd):
        visited = set()
        front = deque([pd])
        while front:
            pd = front.popleft()
            if pd in visited:
                continue
            visited.add(pd)
            for nxt in paths[pd]:
                if nxt is not None:
                    front.append(nxt)
        return len(set(pd.pos for pd in visited))

    while front:
        t, pd, prevpd = heappop(front)
        prevt = visited.get(pd, None)
        if prevt is not None:
            if prevt == t:
                paths[pd].append(prevpd)
            continue
        visited[pd] = t
        paths[pd].append(prevpd)

        pos, d = pd
        if pos == endpos:
            # in my data we can't arrive there from different directions
            if second:
                return unwind(pd)
            return t
        if g[pos + d] == '.':
            npos = posd(pos + d, d)
            if npos not in visited:
                heappush(front, (t + 1, npos, pd))
        npos = posd(pos, d*1j)
        if npos not in visited:
            heappush(front, (t + 1000, npos, pd))
        npos = posd(pos, -d*1j)
        if npos not in visited:
            heappush(front, (t + 1000, npos, pd))
    assert False


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
