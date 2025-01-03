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
        __slots__ = ()
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


def problem17(data, second):
    # if second: return
    _data = split_data('''
Register A: 729
Register B: 0
Register C: 0

Program: 0,1,5,4,3,0
''')
    _data = split_data('''
Register A: 2024
Register B: 0
Register C: 0

Program: 0,3,5,4,3,0''')
    data = iter(data)
    a = int(next(data).split(':')[1])
    b = int(next(data).split(':')[1])
    c = int(next(data).split(':')[1])
    assert not next(data)
    code = list(map(int, next(data).split(':')[1].split(',')))

    def run(a, b, c):
        pc = 0
        res = []

        def combo(arg):
            if arg <= 3:
                return arg
            if arg == 4:
                return a
            if arg == 5:
                return b
            if arg == 6:
                return c
            assert False

        while pc in range(0, len(code)):
            op, arg = code[pc], code[pc + 1]
            # print(f'pc={pc} a={a} b={b} c={c} out={res}')
            pc += 2
            if op == 0: # adv
                a = a // (2 ** combo(arg))
            elif op == 1: # bxl
                b = b ^ arg
            elif op == 2: # bst
                b = combo(arg) % 8
            elif op == 3: # jnz
                if a: pc = arg
            elif op == 4: # bxc
                b = b ^ c
            elif op == 5: # out
                res.append(combo(arg) % 8)
            elif op == 6: # bdv
                b = a // (2 ** combo(arg))
            elif op == 7: # cdv
                c = a // (2 ** combo(arg))
        return res

    def disassemble(code):
        for op, arg in grouper(code, 2):
            ops = ['adv', 'bxl', 'bst', 'jnz', 'bxc', 'out', 'bdv', 'cdv']
            print(f'{ops[op]} {arg}')

    if not second:
        return ','.join(map(str, run(a, b, c)))

    # l'esprit d'escalier
    disassemble(code)

    import z3
    zs = z3.Solver()
    aStart = z3.BitVec('a', 64)
    # aStart = z3.Int('a')
    a, b, c = aStart, 0, 0
    code = code[:2]
    for i, d in enumerate(code):
        # b = a               # bst 4
        b = a ^ 1           # bxl 1
        c = z3.LShR(a, b)   # cdv 5
        b = b ^ c           # bxc 4
        b = b ^ 4           # bxl 4
        a = z3.LShR(a, 3)   # adv 3
        # bx = z3.BitVec(f'b{i}', 64)
        # zs.add(bx == b & 7)
        # zs.add(bx == d)  # out 5
        zs.add(b & 7 == d)  # out 5
        if i != len(code) - 1:
            zs.add(a != 0)  # jnz 0
        else:
            zs.add(a == 0)
    res = []
    while zs.check() == z3.sat:
        solution = zs.model().eval(aStart).as_long()
        res.append(solution)
        zs.add(aStart != solution)
        if len(res) > 5:
            break
    # print(res)
    # print(code)
    # pprint([run(a, 0, 0) for a in res])
    # assert all(run(a, 0, 0) == code for a in res)
    # return min(res)

    pows = [0] * len(code)
    def pows2a(pows):
        a = 0
        p = 1
        for it in reversed(pows):
            a += p * it
            p *= 8
        return a

    def recur(power):
        if power == len(code):
            return pows2a(pows)
        d = code[-1 - power]
        for a1 in range(8):
            if power == 0 and a1 == 0:
                continue
            pows[power] = a1
            a = pows2a(pows)
            res = run(a, 0, 0)
            # print(res)
            if res[-1 - power] == d:
                if a := recur(power + 1):
                    return a
        return 0

    a = recur(0)
    assert run(a, 0, 0) == code
    return a


def problem18(data, second):
    # if second: return
    _data = split_data('''
5,4
4,2
4,5
3,0
2,1
6,3
2,4
1,5
0,6
3,3
2,6
5,1
1,2
5,5
2,5
6,5
1,4
0,4
6,4
1,1
6,1
1,0
0,5
1,6
2,0''')
    data = [int(b) + 1j*int(a) for s in data for a, b in [s.split(',')]]
    width = height = 7
    initial = 12
    width = height = 71
    initial = 1024
    g = np_cidx(np.zeros((height, width), dtype=np.uint8))
    for p in data[:initial]:
        g[p] = 1

    def inside(pos):
        return pos.real in range(width) and pos.imag in range(height)

    def run():
        front = [0j]
        target = width - 1 + (height - 1) * 1j
        visited = set(front)
        steps = 0
        while front:
            new = []
            while front:
                p = front.pop()
                if p == target:
                    return steps
                for d in cdirections4:
                    pn = p + d
                    if inside(pn) and not pn in visited and not g[pn]:
                        new.append(pn)
                        g[pn] = 2
                        visited.add(pn)
            # print(g.arr)
            front = new
            steps += 1
    if not second:
        return run()
    l = 0
    r = len(data)
    while l < r - 1:
        m = (l + r) // 2
        # print(l, r, m)
        g.arr[:] = 0
        for p in data[:m]:
            g[p] = 1
        if run():
            l = m
        else:
            r = m
    # print(data[l - 1 : r + 1])
    return ','.join(str(p) for p in reversed(c2v2(data[l])))


def problem19(data, second):
    # if second: return
    _data = split_data('''
r, wr, b, g, bwu, rb, gb, br

brwrr
bggr
gbbr
rrbgbr
ubwu
bwurrg
brgr
bbrgwb''')
    data = iter(data)
    towels = [s.strip() for s in next(data).split(',')]
    assert not next(data)
    patterns = list(data)

    res = 0
    for pattern in patterns:
        queue = [0]
        visited = set(queue)
        counts = Counter(queue)
        while queue:
            idx = heappop(queue)
            cnt = counts[idx]
            ts = towels
            d = 0
            while ts:
                ts2 = []
                idxnew = idx + d
                for t in ts:
                    if d >= len(t):
                        if idxnew not in visited:
                            heappush(queue, idxnew)
                            visited.add(idxnew)
                        counts[idxnew] += cnt
                    elif idxnew >= len(pattern):
                        continue
                    elif t[d] == pattern[idxnew]:
                        ts2.append(t)
                ts = ts2
                d += 1
        last = counts[len(pattern)]
        # print(last)
        res += last if second else bool(last)
    return res


def problem20(data, second):
    # if second: return
    _data = split_data('''
###############
#...#...#.....#
#.#.#.#.#.###.#
#S#...#.#.#...#
#######.#.#.###
#######.#.#...#
#######.#.###.#
###..E#...#...#
###.#######.###
#...###...#...#
#.#####.#.###.#
#.#...#.#.#...#
#.#.#.#.#.#.###
#...#...#...###
###############''')
    g = ndarray_from_chargrid(data)
    gc = np_cidx(g)
    [startpos] = np.argwhere(g == 'S')
    startpos = v22c(startpos)
    [endpos] = np.argwhere(g == 'E')
    endpos = v22c(endpos)
    gc[startpos] = gc[endpos] = '.'
    height, width = len(g), len(g)

    gr = networkx.Graph()
    for r in range(1, height - 1):
        for c in range(1, width - 1):
            p = r + 1j * c
            if gc[p] == '.':
                for d in cdirections4:
                    p2 = p + d
                    if gc[p2] == '.':
                        gr.add_edge(p, p2)
    d1 = networkx.single_source_dijkstra_path_length(gr, startpos)
    d2 = networkx.single_source_dijkstra_path_length(gr, endpos)
    best = d1[endpos]
    savings = Counter()

    if not second:
        for r in range(1, height - 1):
            for c in range(1, width - 1):
                p = r + 1j * c
                if gc[p] == '#':
                    for d in cdirections4:
                        if gc[p + d] == '.' and gc[p - d] == '.':
                            newtime = d1[p + d] + 2 + d2[p - d]
                            if best - newtime >= 100:
                                savings[best - newtime] += 1
        return sum(savings.values())

    path = networkx.dijkstra_path(gr, startpos, endpos)
    pathset = set(path)

    def check(p, p2, d):
        if p2 not in pathset:
            return
        if p2 in targets:
            return
        targets.add(p2)
        newtime = d1[p] + d + d2[p2]
        if best - newtime >= 100:
            savings[best - newtime] += 1

    for i, p1 in enumerate(path):
        # if not i % 100: print(i)
        targets = set()
        for dr in range(21):
            for dc in range(21 - dr):
                check(p1, p1 + dr + 1j * dc, dr + dc)
                check(p1, p1 + dr - 1j * dc, dr + dc)
                check(p1, p1 - dr + 1j * dc, dr + dc)
                check(p1, p1 - dr - 1j * dc, dr + dc)
    return sum(savings.values())

        # for p2 in itertools.islice(path, i + 1, None):
        #     d = abs(p1.real - p2.real) + abs(p1.imag - p2.imag)
        #     if d <= 20:
        #         newtime = d1[p1] + d + d2[p2]
        #         if best - newtime >= 100:
        #             savings[best - newtime] += 1

    # for r, c in it_product(range(1, height - 1), range(1, width - 1)):
    #     p = r + 1j * c
    #     if gc[p] == '.':
    #         targets = set()
    #         for dr in range(21):
    #             for dc in range(21 - dr):
    #                 check(p, p + dr + 1j * dc, dr + dc)
    #                 check(p, p + dr - 1j * dc, dr + dc)
    #                 check(p, p - dr + 1j * dc, dr + dc)
    #                 check(p, p - dr - 1j * dc, dr + dc)
    # print(sorted(savings.items()))


def problem21(data, second):
    # if second: return
    _data = split_data('''
029A
980A
179A
456A
379A''')
    k1 = ['789', '456', '123', '#0A']
    k1d = {}
    k1a = 3, 2
    for r, c in it_product(range(4), range(3)):
        k1d[k1[r][c]] = (r, c)

    k2 = ['#^A', '<v>']
    k2d = {}
    k2a = 0, 2
    for r, c in it_product(range(2), range(3)):
        k2d[k2[r][c]] = (r, c)

    max_keypad = 25 if second else 2

    @functools.cache
    def move(keypad, r, c, d):
        if keypad:
            kd = k2d
        else:
            kd = k1d

        def move_r(r, rd):
            while rd > r:
                r += 1
                yield 'v'
            while rd < r:
                r -= 1
                yield '^'
        def move_c(c, cd):
            while cd > c:
                c += 1
                yield '>'
            while cd < c:
                c -= 1
                yield '<'

        rd, cd = kd[d]

        def submove(res):
            r, c = k2a
            res2 = 0
            for d in res:
                s, r, c = move(keypad + 1, r, c, d)
                res2 += s
            return res2

        def submove2(rfirst):
            if rfirst:
                if (rd, c) == kd['#']:
                    return ''
            else:
                if (r, cd) == kd['#']:
                    return ''

            res = []
            if rfirst:
                res.extend(move_r(r, rd))
                res.extend(move_c(c, cd))
            else:
                res.extend(move_c(c, cd))
                res.extend(move_r(r, rd))

            res.append('A')
            if keypad == max_keypad:
                return len(res)
            else:
                return submove(res)

        s1 = submove2(True)
        s2 = submove2(False)
        if not s1:
            s = s2
        elif not s2:
            s = s1
        else:
            s = min(s1, s2)
        return s, rd, cd


    def make_moves(keypad, seq):
        r, c = k2a if keypad else k1a
        res = 0
        for d in seq:
            s, r, c = move(keypad, r, c, d)
            res += s
        return res

    r = 0
    for s in data:
        moves = make_moves(0, s)
        c = moves * int(s[:-1])
        r += c
        print(f'{s!r}: {c}')
    return r

# import numba
# @numba.njit
def problem22(data, second):
    return
    # if second: return
#     _data = split_data('''
# 1
# 2
# 3
# 2024''')
    MOD = 16777216
    def step(x):
        y = x * 64
        x = (x ^ y) % MOD
        y = x >> 5
        x = (x ^ y) % MOD
        y = x * 2048
        x = (x ^ y) % MOD
        return x
    res = 0
    prices = []
    for ns in data:
        with numba.objmode(n=numba.int32):
            n = int(ns)
        price = np.zeros(2001, dtype=np.int32)
        price[0] = n % 10
        for i in range(2000):
            n = step(n)
            price[i + 1] = n % 10
        res += n
        prices.append(price)
    if not second:
        return res

    diffs = []
    for p in prices:
        diffs.append(p[1:] - p[:-1])

    cnt = numba.typed.Dict.empty(numba.int64, numba.int64)
    for price, diff in zip(prices, diffs):
        seen = set()
        for i in range(len(diff) - 3):
            k = diff[i] + 100*diff[i + 1] + 100_00*diff[i + 2] + 100_00_00*diff[i + 3]
            if k not in seen:
                cnt[k] = cnt.get(k, 0) + price[i + 4]
                seen.add(k)
    return max(cnt.values())


def problem23(data, second):
    # if second: return
    _data = split_data('''
kh-tc
qp-kh
de-cg
ka-co
yn-aq
qp-ub
cg-tb
vc-aq
tb-ka
wh-tc
yn-cg
kh-ub
ta-co
de-co
tc-td
tb-wq
wh-td
ta-ka
td-qp
aq-cg
wq-ub
ub-vc
de-ta
wq-aq
wq-vc
wh-yn
ka-de
kh-ta
co-tc
wh-qp
tb-vc
td-yn''')
    g = networkx.Graph()
    for s in data:
        a, b = s.split('-')
        g.add_edge(a, b)
    res = set()
    for n in g.nodes:
        if not n.startswith('t'):
            continue
        for e1, e2 in it_product(g.adj[n], g.adj[n]):
            if (e1, e2) in g.edges:
                res.add(tuple(sorted((n, e1, e2))))
    if not second:
        return len(res)
    clique = max(networkx.find_cliques(g), key=len)
    return ','.join(sorted(clique))


def problem24(data, second):
    # if second: return
    _data = split_data('''
x00: 0
x01: 1
x02: 0
x03: 1
x04: 0
x05: 1
y00: 0
y01: 0
y02: 1
y03: 1
y04: 0
y05: 1

x00 AND y00 -> z05
x01 AND y01 -> z02
x02 AND y02 -> z01
x03 AND y03 -> z03
x04 AND y04 -> z04
x05 AND y05 -> z00''')
    data = iter(data)
    signals = {}
    while s := next(data):
        a, b = s.split(': ')
        signals[a] = int(b)
    gates_raw = []
    for s in data:
        a, z = s.split('->')
        x, op, y = a.split()
        z = z.strip()
        gates_raw.append((x, op, y, z))

    gates = {}
    gates_idx = defaultdict(list)
    gates_idx2 = {}
    gates_lst = []
    renames = {}
    rev_renames = {}
    wire_swaps_raw = [] # [('hqk', 'z35'), ('fhc', 'z06'), ('mwh', 'ggt'), ('z11', 'qhj')] if second else []
    wire_swaps = {}
    for a, b in wire_swaps_raw:
        wire_swaps[a] = b
        wire_swaps[b] = a

    def parse_gates():
        gates.clear()
        gates_idx.clear()
        gates_idx2.clear()
        gates_lst.clear()
        for x, op, y, z in gates_raw:
            x = renames.get(x, x)
            y = renames.get(y, y)
            z = wire_swaps.get(z, z)
            z = renames.get(z, z)
            if x > y:
                x, y = y, x
            gates[z] = (op, x, y)
            gates_idx[x].append(z)
            gates_idx[y].append(z)
            gates_lst.append((x, y, op, z))
            gates_idx2[(x, y, op)] = z

    parse_gates()

    OPS = {'OR': lambda x, y: int(x or y),
           'AND': lambda x, y: int(x and y),
           'XOR': lambda x, y: int(x != y),
    }

    def run(signals):
        ping = set(signals)
        seen = set()
        while ping:
            s = ping.pop()
            for z in gates_idx[s]:
                if z in seen:
                    continue
                op, x, y = gates[z]
                if x in signals and y in signals:
                    res = OPS[op](signals[x], signals[y])
                    signals[z] = res
                    seen.add(z)
                    ping.add(z)
        outs = sorted(z for z in signals if z[0] == ('z'))
        return int(''.join(str(signals[c]) for c in reversed(outs)), 2)

    if not second:
        return run(signals)

    INPUT_BITS = 45

    # TESTS = 50
    # rng = random.Random(0)
    # tests = []
    # for _ in range(TESTS):
    #     x = rng.getrandbits(INPUT_BITS)
    #     y = rng.getrandbits(INPUT_BITS)
    #     tests.append((x, y, x + y))

    # def set_input(n, prefix):
    #     for i in range(INPUT_BITS):
    #         signals[f'{prefix}{i:02d}'] = n & 1
    #         n >> 1

    # def count_trailing_zeroes(n):
    #     for res in range(1000):
    #         if not n or (n & 1):
    #             return res
    #         n >>= 1

    # def score_on_tests():
    #     minscore = 1000
    #     for x, y, z in tests:
    #         set_input(x, 'x')
    #         set_input(y, 'y')
    #         rz = run(signals)
    #         score = count_trailing_zeroes(z ^ rz)
    #         minscore = min(minscore, score)
    #     return minscore

    # for depth in range(4):
    #     maxscore = (-1, '', '')
    #     for i in range(len(gates)):
    #         for j in range(i + 1, len(gates)):
    #             z1, z2 = gates_lst[i][3], gates_lst[j][3]
    #             if z1 in wire_swaps or z2 in wire_swaps:
    #                 continue
    #             wire_swaps[z1] = z2
    #             wire_swaps[z2] = z1
    #             parse_gates()
    #             minscore = score_on_tests()
    #             del wire_swaps[z1]
    #             del wire_swaps[z2]
    #             maxscore = max(maxscore, (minscore, z1, z2))
    #         print(depth, i, maxscore)
    #     _, z1, z2 = maxscore
    #     wire_swaps[z1] = z2
    #     wire_swaps[z2] = z1
    # return ','.join(sorted(wire_swaps))

    def gate(x, y, op):
        return (f'x{x:02}', f'y{y:02}', op)

    def rename(a, b):
        assert a not in renames
        assert b not in rev_renames
        renames[a] = b
        rev_renames[b] = a

    for i in range(INPUT_BITS):
        r = gates_idx2[gate(i, i, 'XOR')]
        c = gates_idx2[gate(i, i, 'AND')]
        if r[0] != 'z': rename(r, f'_r{i:02}')
        if c[0] != 'z': rename(c, f'_c{i:02}')

    parse_gates()
    print(len(gates_lst))

    # dump_txt = open('dump.txt', 'w')
    gates_lst.sort()
    for x, y, op, z in gates_lst:
        orig = rev_renames.get(z)
        orig = f' ({orig})' if orig else ''
        s = f'{x} {op:<3} {y} = {z}{orig}'
        # print(s)
        # print(s, file=dump_txt)
    # dump_txt.close()

    return ','.join(sorted(wire_swaps))


def problem25(data, second):
    if second: return
    _data = split_data('''
#####
.####
.####
.####
.#.#.
.#...
.....

#####
##.##
.#.##
...##
...#.
...#.
.....

.....
#....
#....
#...#
#.#.#
#.###
#####

.....
.....
#.#..
###..
###.#
###.#
#####

.....
.....
.....
#....
#.#..
#.#.#
#####''')

    arr = []
    locks = []
    keys = []

    def parse_arr():
        if not arr:
            return
        profile = []
        for col in range(len(arr[0])):
            cnt = 0
            for row in range(1, len(arr)):
                if arr[row][col] != arr[0][col]:
                    break
                cnt += 1
            profile.append(cnt)
        if arr[0][0]=='#':
            locks.append(profile)
        else:
            keys.append(profile)
        arr.clear()

    for s in data:
        if not s:
            parse_arr()
        else:
            arr.append(s)
    parse_arr()

    return sum(all(l <= k for l, k in zip(ll, kk)) for ll, kk in it_product(locks, keys))


##########

def problem(data, second):
    if second: return
    data = split_data(''' ''')


##########

if __name__ == '__main__':
    print('Hello')
    solve_latest()
    # solve_latest(17)
    # solve_all()
