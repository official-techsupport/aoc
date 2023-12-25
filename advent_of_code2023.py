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
    _data = split_data('''
...........
.S-------7.
.|F-----7|.
.||OOOOO||.
.||OOOOO||.
.|L-7OF-J|.
.|II|O|II|.
.L--JOL--J.
.....O.....''')
    g = networkx.DiGraph()

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
            elif c == '-':
                add_edge(row, col, 1)
                add_edge(row, col, 3)
            elif c == 'L':
                add_edge(row, col, 0)
                add_edge(row, col, 1)
            elif c == 'J':
                add_edge(row, col, 0)
                add_edge(row, col, 3)
            elif c == '7':
                add_edge(row, col, 2)
                add_edge(row, col, 3)
            elif c == 'F':
                add_edge(row, col, 2)
                add_edge(row, col, 1)
            elif c == 'S':
                start = (row, col)
            elif c in '.IO':
                pass
            else:
                assert False

    gclean = networkx.Graph()
    for src, dst in g.edges:
        if (dst, src) in g.edges:
            gclean.add_edge(src, dst)

    assert len(g.in_edges(start)) == 2
    for e in g.in_edges(start):
        gclean.add_edge(*e)

    g = gclean

    for c in networkx.connected_components(g):
        if start in c:
            break
    else:
        assert False

    if not second:
        return networkx.eccentricity(g.subgraph(c), start)

    m = ndarray_from_chargrid(data)
    for i, row in enumerate(m):
        for j in range(len(row)):
            if (i, j) not in c:
                row[j] = '.'
            if row[j] == 'S':
                # fixup manually depending on input
                # row[j] = 'F'
                row[j] = '|'

    counter = 0
    for row in m:
        inside = False
        prev = None
        for c in row:
            if c == '.' and inside:
                counter += 1
            elif c == '|':
                inside = not inside
            elif c in 'FL':
                prev = c
            elif c == '7':
                if prev == 'L':
                    inside = not inside
            elif c == 'J':
                if prev == 'F':
                    inside = not inside

    return counter


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


def problem12(data, second):
    _data = split_data('''
???.### 1,1,3
.??..??...?##. 1,1,3
?#?#?#?#?#?#?#? 1,3,1,6
????.#...#... 4,1,1
????.######..#####. 1,6,5
?###???????? 3,2,1''')
    def count_ways(pat, groups):
        @functools.cache
        def rec(pat_idx, group_idx):
            if pat_idx >= len(pat):
                if group_idx < len(groups):
                    return 0
                return 1

            ways = 0
            if pat[pat_idx] in '?.':
                # skip one
                ways = rec(pat_idx + 1, group_idx)

            # start group here
            if group_idx >= len(groups):
                return ways
            glen = groups[group_idx]
            if pat_idx + glen > len(pat):
                return ways
            for i in range(glen):
                if pat[pat_idx + i] not in '?#':
                    return ways
            if pat_idx + glen < len(pat):
                if pat[pat_idx + glen] not in '?.':
                    return ways
            return ways + rec(pat_idx + glen + 1, group_idx + 1)
        return rec(0, 0)

    total = 0
    for s in data:
        pat, groups = s.split()
        groups = [int(g) for g in groups.split(',')]
        if second:
            pat = '?'.join([pat] * 5)
            groups = groups * 5
        ways = count_ways(pat, groups)
        total += ways
        # print(pat, groups, ways)
    return total


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

    boxes = [{} for _ in range(256)]
    for s in data:
        if s.endswith('-'):
            label = s[:-1]
            h = holiday_ash(label)
            box = boxes[h]
            box.pop(label, None)
        else:
            label, f = s.split('=')
            h = holiday_ash(label)
            box = boxes[h]
            box[label] = f

    res = 0
    for i, box in enumerate(boxes):
        for j, f in enumerate(box.values()):
            res += (i + 1) * (j + 1) * int(f)
    return res


def problem16(data, second):
    _data = split_data(r'''
.|...\....
|.-.\.....
.....|-...
........|.
..........
.........\
..../.\\..
.-.-/..|..
.|....-|.\
..//.|....''')
    m = ndarray_from_chargrid(data)
    width, height = len(m[0]), len(m)

    def run(start):
        front = deque([start])
        visited = set(front)
        dirs = ((0, 1), (1, 0), (0, -1), (-1, 0)) # start right, go cw
        # >0 V1 <2 ^3

        while front:
            row, col, dir = front.popleft()
            c = m[row, col]

            def add(row, col, dir):
                dr, dc = dirs[dir]
                row += dr
                col += dc
                if not (0 <= row < height and 0 <= col < width):
                    return
                key = (row, col, dir)
                if key in visited:
                    return
                front.append(key)
                visited.add(key)

            if c == '.':
                add(row, col, dir)
            elif c == '\\':
                # >0 V1 <2 ^3
                dir = {0: 1, 1: 0, 2: 3, 3: 2}[dir]
                add(row, col, dir)
            elif c == '/':
                dir = {0: 3, 1: 2, 2: 1, 3: 0}[dir]
                add(row, col, dir)
            elif c in '|-':
                if (c == '|' and dir in (1, 3)) or (c == '-' and dir in (0, 2)):
                    add(row, col, dir)
                else:
                    add(row, col, (dir + 1) % 4)
                    add(row, col, (dir + 3) % 4)
            else:
                assert False

        mapvisited = {(row, col) for row, col, dir in visited}
        return len(mapvisited)

    starts = []
    if not second:
        starts.append((0, 0, 0))
    else:
        for row in range(height):
            starts.append((row, 0, 0))
            starts.append((row, width - 1, 2))
        for col in range(width):
            starts.append((0, col, 1))
            starts.append((height - 1, col, 3))

    return max(run(s) for s in starts)


def problem17(data, second):
    _data = split_data('''
2413432311323
3215453535623
3255245654254
3446585845452
4546657867536
1438598798454
4457876987766
3637877979653
4654967986887
4564679986453
1224686865563
2546548887735
4322674655533''')

    _data = split_data('''
111111111111
999999999991
999999999991
999999999991
999999999991''')

    m = ndarray_from_chargrid(data)
    m = m.astype(int)
    width, height = len(m[0]), len(m)

    # dir: 0> 1^ 2< 3V
    dirs = ((0, 1), (-1, 0), (0, -1), (1, 0))

    start_keys = [((0, 0), 0, 0), ((0, 0), 3, 0)]
    visited = {k: 0 for k in start_keys}
    front = deque(visited.keys())

    def add(score, pos, dir, rl):
        dir %= 4
        pos = addv2(pos, dirs[dir])
        row, col = pos
        if not (0 <= row < height and 0 <= col < width):
            return
        key = pos, dir, rl
        score += m[pos]
        if visited.get(key, 999999999) <= score:
            return
        visited[key] = score
        front.append(key)

    while front:
        pos, dir, rl = key = front.popleft()
        score = visited[key]
        if not second or rl >= 3:
            add(score, pos, dir - 1, 0)
            add(score, pos, dir + 1, 0)
        if rl < (9 if second else 2):
            add(score, pos, dir, rl + 1)

    return min(score for ((row, col), dir, rl), score in visited.items()
               if row == height - 1 and col == width - 1 and rl >= (3 if second else 0))


def problem18(data, second):
    _data = split_data('''
R 6 (#70c710)
D 5 (#0dc571)
L 2 (#5713f0)
D 2 (#d2c081)
R 2 (#59c680)
D 2 (#411b91)
L 5 (#8ceee2)
U 2 (#caa173)
L 1 (#1b58a2)
U 2 (#caa171)
R 2 (#7807d2)
U 3 (#a77fa3)
L 2 (#015232)
U 2 (#7a21e3)''')
    if second: return

    dd = []
    for s in data:
        d, l, c = s.split()
        dd.append((d, int(l)))

    dd2 = []
    for d, l in dd:
        if d == 'R':
            x = (0, l)
        elif d == 'D':
            x = (l, 0)
        elif d == 'L':
            x = (0, -l)
        elif d == 'U':
            x = (-l, 0)
        else:
            assert False
        dd2.append(x)

    x, y = 0, 0
    minx, maxx, miny, maxy = 999999, -999999, 999999, -999999
    for dy, dx in dd2:
        x += dx
        y += dy
        minx = min(minx, x)
        maxx = max(maxx, x)
        miny = min(miny, y)
        maxy = max(maxy, y)

    maxx += 1
    maxy += 1

    m = np.ndarray((maxy - miny, maxx - minx), dtype='U1')
    m[:] = '.'
    x, y = -minx, -miny
    for dy, dx in dd2:
        while dx:
            m[y, x] = '#'
            x += np.sign(dx)
            dx -= np.sign(dx)
        while dy:
            m[y, x] = '#'
            y += np.sign(dy)
            dy -= np.sign(dy)

    front = deque([(-minx + 1, -miny + 1)])
    while front:
        x, y = front.popleft()
        if m[y, x] != '.':
            continue
        m[y, x] = '#'
        for dx, dy in ((0, 1), (-1, 0), (0, -1), (1, 0)):
            front.append((x + dx, y + dy))

    print(m)
    # print('\n'.join(''.join(c) for c in s for s in m))

    return np.sum(m == '#')


def problem19(data, second):
    _data = split_data('''
px{a<2006:qkq,m>2090:A,rfg}
pv{a>1716:R,A}
lnx{m>1548:A,A}
rfg{s<537:gd,x>2440:R,A}
qs{s>3448:A,lnx}
qkq{x<1416:A,crn}
crn{x>2662:A,R}
in{s<1351:px,qqz}
qqz{s>2770:qs,m<1801:hdj,R}
gd{a>3333:R,R}
hdj{m>838:A,pv}

{x=787,m=2655,a=1222,s=2876}
{x=1679,m=44,a=2067,s=496}
{x=2036,m=264,a=79,s=2244}
{x=2461,m=1339,a=466,s=291}
{x=2127,m=1623,a=2188,s=1013}''')
    if second: return

    def partition(s):
        m = next(re.finditer('<|>', s))
        idx = m.start()
        return s[0:idx], s[idx], s[idx + 1:]

    rules = {}
    for idx, s in enumerate(data):
        if not s:
            break
        name, rest = s.split('{')
        rest = rest.strip('}')
        conds = rest.split(',')
        rule = []
        rules[name] = rule
        for c in conds:
            if ':' in c:
                cond, dst = c.split(':')
                var, cond, val = partition(cond)
            else:
                dst = c
                var, cond, val = '1', '>', '0'
            assert cond in '<>'
            if cond == '<':
                cond = operator.lt
            else:
                cond = operator.gt
            rule.append((var, cond, int(val), dst))

    # print(rules)
    items = []
    for s in data[idx + 1:]:
        s = s.strip('{}')
        it = {}
        items.append(it)
        for ss in s.split(','):
            k, v = ss.split('=')
            it[k] = int(v)

    accepted = []
    for part in items:
        part['1'] = 1
        cur = 'in'
        while cur:
            if cur == 'A':
                accepted.append(part)
                break
            elif cur == 'R':
                break

            flow = rules[cur]
            for var, cond, val, dst in flow:
                if cond(part[var], val):
                    cur = dst
                    break
            else:
                assert False

    res = 0
    for it in accepted:
        del it['1']
        res += sum(it.values())
    if not second:
        return res


def problem20(data, second):
    _data = split_data('''
broadcaster -> a, b, c
%a -> b
%b -> c
%c -> inv
&inv -> a''')
    _data = split_data('''
broadcaster -> a
%a -> inv, con
&inv -> b
%b -> con
&con -> output''')

    class Module:
        type: str
        outputs: list[str]
        inputs: dict
        state = 0

    modules = {}
    for s in data:
        name, _, outputs = s.partition(' -> ')
        assert outputs
        m = Module()
        if name[0] in '%&':
            m.type = name[0]
            modules[name[1:]] = m
        else:
            assert name == 'broadcaster'
            m.type = 'b'
            modules[name] = m
        m.outputs = [s.strip() for s in outputs.split(',')]
        m.inputs = {}

    extras = []
    for name, m in modules.items():
        for m2 in m.outputs:
            if m2 not in modules:
                extras.append(m2)
            else:
                modules[m2].inputs[name] = 0

    # print(extras)
    for mname in extras:
        modules[mname] = m = Module()
        m.type = 'o'
        m.outputs = []
        m.inputs = {}

    pulsescnt = defaultdict(int)

    for press in range(100000000 if second else 1000):
        signals = deque()
        signals.append(('button', 'broadcaster', 0))
        while signals:
            src, dst, level = signals.popleft()
            # print(f'{src} -{"high" if level else "low"}-> {dst}')
            pulsescnt[level] += 1
            if second and dst == 'rx' and not level:
                return press + 1

            m = modules[dst]
            if m.type == '%':
                if level: continue
                m.state = not m.state
                for out in m.outputs:
                    signals.append((dst, out, m.state))
            elif m.type == '&':
                # print('!!!', m.inputs, level)
                m.inputs[src] = level
                outlevel = not all(m.inputs.values())
                for out in m.outputs:
                    signals.append((dst, out, outlevel))
            elif m.type == 'b':
                for out in m.outputs:
                    signals.append((dst, out, level))
            elif m.type == 'o':
                pass
            else:
                assert False

    print(pulsescnt)
    return pulsescnt[0] * pulsescnt[1]


def problem21(data, second):
    _data = split_data('''...........
.....###.#.
.###.##..#.
..#.#...#..
....#.#....
.##..S####.
.##..#...#.
.......##..
.##.#.####.
.##..##.##.
...........''')
    m = ndarray_from_chargrid(data)
    width, height = len(m[0]), len(m)
    [[start_row], [start_col]] = np.where(m == 'S')

    if not second:
        front = [(start_row, start_col)]
        for step in range(64):
            new_front = []
            visited = set()
            while front:
                row, col = pos = front.pop()

                for dr, dc in ((0, 1), (-1, 0), (0, -1), (1, 0)):
                    nrow = row + dr
                    ncol = col + dc
                    if not (0 <= nrow < height and 0 <= ncol < width):
                        continue
                    npos = (nrow, ncol)
                    if npos in visited:
                        continue
                    if m[npos] == '#':
                        continue
                    new_front.append(npos)
                    visited.add(npos)
            front = new_front
            m1 = np.copy(m)
            for rc in visited:
                m1[rc] = 'O'
            # print(step)
            # print(m1)
            # print()
        return len(visited)

    ### Second
    front = [(start_row, start_col)]
    visited = {}
    for step in range(6400):
        new_front = []
        while front:
            row, col = front.pop()

            for dr, dc in ((0, 1), (-1, 0), (0, -1), (1, 0)):
                nrow = row + dr
                ncol = col + dc
                npos = (nrow, ncol)
                if not (0 <= nrow < height and 0 <= ncol < width):
                    continue
                if npos in visited:
                    continue
                if m[npos] == '#':
                    continue
                visited[npos] = step + 1
                new_front.append(npos)
        front = new_front
    mv = np.zeros_like(m, int)
    for pos, step in visited.items():
        mv[pos] = step
    print(m.shape)
    print(26501365 - (26501365 // 131 * 131))


def problem22(data, second):
    data = split_data('''
1,0,1~1,2,1
0,0,2~2,0,2
0,2,3~2,2,3
0,0,4~0,2,4
2,0,5~2,2,5
0,1,6~2,1,6
1,1,8~1,1,9''')
    if second: return

    bricks = []
    for s in data:
        xyz1, _, xyz2 = s.partition('~')
        def parse(s):
            return list(map(int, s.split(',')))
        bricks.append((parse(xyz1), parse(xyz2)))

    # for (x1,y1,z1), (x2,y2,z2) in bricks:
    #     assert x1 < 10, x2 < 10

    def sort_key(b):
        return min(b[0][2], b[1][2])
    bricks.sort(key=sort_key)

    m = np.zeros((3, 3, 10), np.int8)

    def brick_range(a, b):
        if b > a:
            return range(a, b + 1)
        return range(b, a + 1)

    for idx, ((x1,y1,z1), (x2,y2,z2)) in enumerate(bricks):
        # print(m)
        h = 0
        for x in brick_range(x1, x2):
            for y in brick_range(y1, y2):
                pos_h = np.where(m[x, y])[0].max(initial=-1) + 1
                h = max(h, pos_h)


        minz = min(z1, z2)
        z1 = bricks[idx][0][2] = bricks[idx][0][2] + h - minz
        z2 = bricks[idx][1][2] = bricks[idx][1][2] + h - minz

        for x in brick_range(x1, x2):
            for y in brick_range(y1, y2):
                for z in brick_range(z1, z2):
                    assert not m[x, y, z]
                    m[x, y, z] = idx + 1

    print(m)
    return
    # print(bricks)
    disintegratable = 0
    supports = defaultdict(set)
    for idx, ((x1,y1,z1), (x2,y2,z2)) in enumerate(bricks):
        if idx == 6:
            ...
        minz = min(z1, z2)
        if minz == 0:
            continue
        for x in brick_range(x1, x2):
            for y in brick_range(y1, y2):
                below = x, y, minz - 1
                if m[below]:
                    supports[m[below]].add(idx)
    print('bricks', bricks)
    print('supports', supports)

    for b, s in supports.items():
        print(b, s)

    disintegratable = len(bricks) - sum(1 for x in supports.values() if len(x) >= 2)
    return disintegratable


def problem24(data, second):
    _data = split_data('''19, 13, 30 @ -2,  1, -2
18, 19, 22 @ -1, -1, -2
20, 25, 34 @ -2, -2, -4
12, 31, 28 @ -1, -2, -1
20, 19, 15 @  1, -5, -3''')
    positions = np.zeros((len(data), 3), object)
    directions = np.copy(positions)
    for i, s in enumerate(data):
        ss = [s1.strip(',') for s1 in s.split()]
        positions[i][0] = int(ss[0])
        positions[i][1] = int(ss[1])
        positions[i][2] = int(ss[2])
        directions[i][0] = int(ss[4])
        directions[i][1] = int(ss[5])
        directions[i][2] = int(ss[6])

    def line_intersection(pos1, dir1, pos2, dir2):
        xdiff = dir1[0], dir2[0]
        ydiff = dir1[1], dir2[1]

        def det(a, b):
            return a[0] * b[1] - a[1] * b[0]

        div = det(xdiff, ydiff)
        if div == 0:
            return None, None

        d = det(pos1, addv2(pos1, dir1)), det(pos2, addv2(pos2, dir2))
        x = det(d, xdiff) / div
        y = det(d, ydiff) / div
        return -x, -y

    if not second:
        within = 0
        ta1, ta2 = 200000000000000, 400000000000000
        # ta1, ta2 = 7, 27
        for i in range(len(positions)):
            for j in range(i + 1, len(positions)):
                ix, iy = line_intersection(positions[i], directions[i], positions[j], directions[j])
                if ix is not None:
                    future1 = (ix - positions[i][0]) * directions[i][0] + (iy - positions[i][1]) * directions[i][1]
                    future2 = (ix - positions[j][0]) * directions[j][0] + (iy - positions[j][1]) * directions[j][1]
                else:
                    future1 = 0
                    future2 = 0
                # print(positions[i], positions[j])
                # print(ix, iy, future1 > 0 and future2 > 0)
                if future1 > 0 and future2 > 0 and ta1 <= ix <= ta2 and ta1 <= iy <= ta2:
                    within += 1
                    # print('yayyyy')

        return within

    import z3
    x, y, z = z3.Ints('x y z')
    dx, dy, dz = z3.Ints('dx dy dz')
    zs = z3.Solver()
    for i in range(len(positions)):
        t = z3.Int(f't{i}')
        zs.add(t >= 0)
        zs.add(positions[i][0] + t * directions[i][0] == x + t * dx)
        zs.add(positions[i][1] + t * directions[i][1] == y + t * dy)
        zs.add(positions[i][2] + t * directions[i][2] == z + t * dz)

    assert zs.check() == z3.sat
    return zs.model().eval(x + y + z).as_long()


def problem25(data, second):
    _data = split_data('''
jqt: rhn xhk nvd
rsh: frs pzl lsr
xhk: hfx
cmg: qnr nvd lhk bvb
rhn: xhk bvb hfx
bvb: xhk hfx
pzl: lsr hfx nvd
qnr: nvd
ntq: jqt hfx bvb xhk
nvd: lhk
lsr: lhk
rzs: qnr cmg lsr rsh
frs: qnr lhk lsr''')
    if second: return
    nx = networkx
    g = nx.Graph()
    for s in data:
        n1, ns = s.split(':')
        for n2 in ns.split():
            g.add_edge(n1, n2)
    # nx.draw(g, with_labels=True)
    # import matplotlib.pyplot as plt
    # plt.show()
    assert nx.number_connected_components(g) == 1
    g.remove_edge('mtq', 'jtr')
    g.remove_edge('pzq', 'rrz')
    g.remove_edge('ddj', 'znv')
    g1, g2 = nx.connected_components(g)
    return len(g1) * len(g2)
    return None


#########

def problem(data, second):
    data = split_data(''' ''')
    if second: return

    return None


##########

if __name__ == '__main__':
    print('Hello')
    # solve_latest()
    # solve_all()
    solve_latest(12)
