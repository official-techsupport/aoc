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

def problem3(data, second):
#     data = split_data('''vJrwpWtwJgWrhcsFMMfFFhFp
# jqHRNqRjqzjGDLGLrsFMfFZSrLrFZsSL
# PmmdzqPrVvPwwTWBwg
# wMqvLMZHhHMvwLHjbvcjnnSBnvTQFn
# ttgJtRGJQctTZtZT
# CrZsJsPPZsGzwwsLwLmpwMDw''')

    def score(c):
        o = ord(c)
        if ord('a') <= o <= ord('z'):
            return o - ord('a') + 1
        if ord('A') <= o <= ord('Z'):
            return o - ord('A') + 27
        assert False

    res = 0
    if second:
        for s1, s2, s3 in grouper(data, 3):
            common = frozenset(s1) & frozenset(s2) & frozenset(s3)
            common, = common
            res += score(common)
    else:
        for s in data:
            assert not len(s) % 2
            half = len(s)//2
            p1, p2 = s[ : half], s[half : ]
            common = frozenset(p1) & frozenset(p2)
            common, = common
            res += score(common)

    return res


def problem4(data, second):
    _data = split_data('''2-4,6-8
2-3,4-5
5-7,7-9
2-8,3-7
6-6,4-6
2-6,4-8''')
    rt = ReTokenizer('{int}-{int},{int}-{int}')
    res = 0
    for a1, a2, b1, b2 in rt.match_all(data):
        # print(a1, a2, b1, b2, res)
        if b1 >= a1 and b2 <= a2:
            res += 1
            continue
        if a1 >= b1 and a2 <= b2:
            res += 1
            continue
        if not second:
            continue

        if b1 <= a1 and b2 >= a1:
            res += 1
            continue
        if a1 <= b1 and a2 >= b1:
            res += 1
            continue

    return res


def problem5(data, second):
    data = get_raw_data()
    _data = '''\
    [D]....
[N] [C]....
[Z] [M] [P]
 1   2   3

move 1 from 2 to 1
move 3 from 1 to 3
move 2 from 2 to 1
move 1 from 1 to 2
'''
    data = data.split('\n')
    cnt, remainder = divmod(len(data[0]) + 1, 4)
    assert remainder == 0

    def parse_stacks():
        stacks = [[] for _ in range(cnt)]
        for i, s in enumerate(data):
            for j, c in enumerate(s[1::4]):
                if c == '1':
                    for stack in stacks:
                        stack.reverse()
                    return stacks, i
                if c not in ' .':
                    stacks[j].append(c)

    stacks, i = parse_stacks()
    # print(stacks)

    for cnt, fro, to in ReTokenizer('move {int} from {int} to {int}').match_all(data[i + 2 : -1]):
        # print(cnt, fro, to)
        fro -= 1
        to -= 1
        if not second:
            for _ in range(cnt):
                stacks[to].append(stacks[fro].pop())
        else:
            stacks[to].extend(stacks[fro][-cnt : ])
            del stacks[fro][-cnt : ]
        # pprint(stacks)
    res = ''.join(stack[-1] for stack in stacks)
    return res


def problem6(data, second):
    _data = 'zcfzfwzzqfrljwzlrfnpqdbhtmscgvjw'
    buf = []
    if second:
        span = 14
    else:
        span = 4
    for i, c in enumerate(data):
        buf.append(c)
        if len(buf) > span:
            del buf[0]
        if len(buf) == span and len(set(buf)) == span:
            return i + 1
    assert False


def problem7(data, second):
    _data = split_data('''$ cd /
$ ls
dir a
14848514 b.txt
8504156 c.dat
dir d
$ cd a
$ ls
dir e
29116 f
2557 g
62596 h.lst
$ cd e
$ ls
584 i
$ cd ..
$ cd ..
$ cd d
$ ls
4060174 j
8033020 d.log
5626152 d.ext
7214296 k''')
    tree = {}
    current = tree
    for s in data:
        def consume(token):
            nonlocal s
            if s.startswith(token):
                _, _, s = s.partition(' ')
                return True

        if consume('$'):
            if consume('ls'):
                assert s == ''
            elif consume('cd'):
                if s == '/':
                    current = tree
                else:
                    current = current[s]
                    assert not isinstance(current, int)
            else:
                assert False
        else:
            size, name = s.split()
            if size == 'dir':
                current.setdefault(name, {'..': current})
            else:
                current[name] = int(size)

    answer1 = 0
    dirs = []
    def rec1(it):
        nonlocal answer1
        size = 0
        for k, v in it.items():
            if isinstance(v, int):
                size += v
            elif k != '..':
                size += rec1(v)
        if size <= 100_000:
            answer1 += size
        dirs.append(size)
        return size
    unused = 70000000 - rec1(tree)
    if not second:
        return answer1
    # print(unused)
    to_delete = 30000000 - unused
    # print(to_delete)
    return min(size for size in dirs if size >= to_delete)


def problem8(data, second):
    _data = split_data('''30373
25512
65332
33549
35390''')
    h = len(data)
    w = len(data[0])
    assert all(len(s) == w for s in data)

    vis = set()

    def dix(pos):
        return data[pos[0]][pos[1]]

    def go(pos, dir, cnt):
        tallest = None
        for _ in range(cnt):
            h = dix(pos)
            if not tallest or tallest < h:
                vis.add(pos)
                tallest = h
            pos = addv2(pos, dir)

    def go2(pos, dir):
        myh = dix(pos)
        for i in range(1000):
            pos = addv2(pos, dir)
            if not (0 <= pos[0] < h) or not (0 <= pos[1] < w):
                return i
            if myh <= dix(pos):
                return i + 1
        assert False

    for i in range(h):
        go((i, 0), (0, 1), w)
        go((i, w - 1), (0, -1), w)

    for i in range(w):
        go((0, i), (1, 0), h)
        go((h - 1, i), (-1, 0), h)

    if not second:
        return len(vis)

    scores = []
    for i in range(1, h - 1):
        for j in range(1, w - 1):
            pos = (i, j)
            scores.append(
                go2(pos, (0, 1)) *
                go2(pos, (0, -1)) *
                go2(pos, (1, 0)) *
                go2(pos, (-1, 0)))

    return max(scores)


def problem9(data, second):
    _data = split_data('''R 4
U 4
L 3
D 1
R 4
D 1
L 5
R 2''')
    if second:
        _data = split_data('''R 4
R 5
U 8
L 8
D 3
R 17
D 10
L 25
U 20''')
    snek = [0j] * (10 if second else 2)
    visited = set([0j])

    def moveh(c, h):
        return h + {'U': 1j, 'R': 1, 'D': -1j, 'L': -1}[c]

    def _movet(h, t):
        diff = h - t
        if abs(diff.real) > 1 and abs(diff.imag) > 1:
            return (t.real + diff.real / 2) + 1j * (t.imag + diff.imag / 2)
        elif abs(diff.real) > 1:
            return (t.real + diff.real / 2) + 1j * h.imag
        elif abs(diff.imag) > 1:
            return h.real                   + 1j * (t.imag + diff.imag / 2)
        return t

    # thx to hbtz
    def movet(h, t):
        d = h - t
        if abs(d.real) > 1 or abs(d.imag) > 1:
            return t + d.real / max(1, abs(d.real)) + 1j * d.imag / max(1, abs(d.imag))
        return t

    for s in data:
        dir, dist = s.split()
        for _ in range(int(dist)):
            snek[0] = moveh(dir, snek[0])
            for i in range(len(snek) - 1):
                snek[i + 1] = movet(snek[i], snek[i + 1])
            visited.add(snek[-1])
            # print(dir, snek)

    return len(visited)


def problem10(data, second):
    _data = split_data('''addx 15
addx -11
addx 6
addx -3
addx 5
addx -1
addx -8
addx 13
addx 4
noop
addx -1
addx 5
addx -1
addx 5
addx -1
addx 5
addx -1
addx 5
addx -1
addx -35
addx 1
addx 24
addx -19
addx 1
addx 16
addx -11
noop
noop
addx 21
addx -15
noop
noop
addx -3
addx 9
addx 1
addx -3
addx 8
addx 1
addx 5
noop
noop
noop
noop
noop
addx -36
noop
addx 1
addx 7
noop
noop
noop
addx 2
addx 6
noop
noop
noop
noop
noop
addx 1
noop
noop
addx 7
addx 1
noop
addx -13
addx 13
addx 7
noop
addx 1
addx -33
noop
noop
noop
addx 2
noop
noop
noop
addx 8
noop
addx -1
addx 2
addx 1
noop
addx 17
addx -9
addx 1
addx 1
addx -3
addx 11
noop
noop
addx 1
noop
addx 1
noop
noop
addx -13
addx -19
addx 1
addx 3
addx 26
addx -30
addx 12
addx -1
addx 3
addx 1
noop
noop
noop
addx -9
addx 18
addx 1
addx 2
noop
noop
addx 9
noop
noop
noop
addx -1
addx 2
addx -37
addx 1
addx 3
noop
addx 15
addx -21
addx 22
addx -6
addx 1
noop
addx 2
addx 1
noop
addx -10
noop
noop
addx 20
addx 1
addx 2
addx 2
addx -6
addx -11
noop
noop
noop''')
    x = 1
    cycle = 1
    probes = [20, 60, 100, 140, 180, 220]
    res = {}
    img = np.full(6*40, '.')
    def draw():
        if cycle in probes:
            res[cycle] = x
        c = cycle - 1
        if abs(c % 40 - x) <= 1:
            img[c] = '#'

    for s in data:
        draw()
        if s == 'noop':
            pass
        else:
            cmd, val = s.split()
            assert cmd == 'addx'
            val = int(val)
            cycle += 1
            draw()
            x += val
        cycle += 1
    assert len(res) == len(probes)
    if not second:
        return sum(k * v for k, v in res.items())
    return 'PGHFGLUG'
    img = img.reshape((6, 40))
    for line in img:
        print(''.join(line))


def problem11(data, second):
    _data = split_data('''Monkey 0:
  Starting items: 79, 98
  Operation: new = old * 19
  Test: divisible by 23
    If true: throw to monkey 2
    If false: throw to monkey 3

Monkey 1:
  Starting items: 54, 65, 75, 74
  Operation: new = old + 6
  Test: divisible by 19
    If true: throw to monkey 2
    If false: throw to monkey 0

Monkey 2:
  Starting items: 79, 60, 97
  Operation: new = old * old
  Test: divisible by 13
    If true: throw to monkey 1
    If false: throw to monkey 3

Monkey 3:
  Starting items: 74
  Operation: new = old + 3
  Test: divisible by 17
    If true: throw to monkey 0
    If false: throw to monkey 1''')

    monkeys = []
    @dataclass
    class Monkey:
        items : List[int] = dataclass_field(default_factory=list)
        operation = None
        test = 0
        test_true = 0
        test_false= 0
        inspected = 0

    def consume(s, prefix):
        assert s.startswith(prefix)
        return s[len(prefix):]

    for s in data:
        op, args = s.split(':')
        # print(f'{op!r}: {args!r}')
        if op.startswith('Monkey'):
            assert not args
            _, arg = op.split()
            assert int(arg) == len(monkeys)
            monke = Monkey()
            monkeys.append(monke)
        elif op == 'Starting items':
            monke.items = [int(s.strip()) for s in args.split(',')]
        elif op == 'Operation':
            args = consume(args, ' new = old ')
            op = args[0]
            assert op in '*+'
            arg = args[1:].strip()
            if arg != 'old':
                arg = int(arg)
            if op == '+':
                monke.operation = lambda x, arg=arg: x + (x if arg == 'old' else arg)
            else:
                monke.operation = lambda x, arg=arg: x * (x if arg == 'old' else arg)
        elif op == 'Test':
            args = consume(args, ' divisible by ')
            monke.test = int(args)
        elif op == 'If true':
            args = consume(args, ' throw to monkey ')
            monke.test_true = int(args)
        elif op == 'If false':
            args = consume(args, ' throw to monkey ')
            monke.test_false = int(args)
        else:
            assert False

    # pprint(monkeys)

    modulo = functools.reduce(operator.mul, [m.test for m in monkeys], 1)

    for round in range(10_000 if second else 20):
        for i, m in enumerate(monkeys):
            # print(f'Monke {i}')
            items = m.items
            m.items = []
            for it in items:
                m.inspected += 1
                # print(f'inspect {it}')
                it = m.operation(it)
                # print(f'worry to {it}')
                if not second:
                    it //= 3
                it %= modulo
                # print(f'worry to {it}')
                target = m.test_false if it % m.test else m.test_true
                # print(f'target {target}')
                monkeys[target].items.append(it)
        # pprint(monkeys)

    top = [-m.inspected for m in monkeys]
    heapify(top)
    a, b = heappop(top), heappop(top)
    return a * b


def problem12(data, second):
    _data = split_data('''Sabqponm
abcryxxl
accszExk
acctuvwj
abdefghi''')

    g = networkx.DiGraph()

    def translate(c):
        if c == 'S': return ord('a')
        if c == 'E': return ord('z')
        return ord(c)

    def get(p):
        i, j = p
        if 0 <= i < len(data):
            if 0 <= j < len(data[i]):
                return translate(data[i][j])
        return -5

    start, goal = [], None

    for i, s in enumerate(data):
        for j, c in enumerate(s):
            p = (i, j)
            if c == 'S' or (c == 'a' and second):
                start.append(p)
            elif c == 'E':
                assert not goal
                goal = p

            for dir in directions4:
                p2 = addv2(p, dir)
                if get(p2) - get(p) <= 1:
                    g.add_edge(p, p2)

    found, visited = bfs_full(start, lambda x: x == goal, lambda x: g.neighbors(x))
    assert found
    p = bfs_extract_path(goal, visited)
    return len(p) - 1


def problem13(data, second):
    _data = split_data('''[1,1,3,1,1]
[1,1,5,1,1]

[[1],[2,3,4]]
[[1],4]

[9]
[[8,7,6]]

[[4,4],4,4]
[[4,4],4,4,4]

[7,7,7,7]
[7,7,7]

[]
[3]

[[[]]]
[[]]

[1,[2,[3,[4,[5,6,7]]]],8,9]
[1,[2,[3,[4,[5,6,0]]]],8,9]''')
    # if second: return
    def compare(first, second):
        if first is None:
            return -1
        if second is None:
            return 1

        if isinstance(first, int) and isinstance(second, int):
            return first - second

        if isinstance(first, int):
            first = [first]
        if isinstance(second, int):
            second = [second]

        for a, b in itertools.zip_longest(first, second):
            c = compare(a, b)
            if c != 0:
                return c

        return 0

    if not second:
        res = 0
        for i, (first, second) in enumerate(grouper(data, 2)):
            first, second = ast.literal_eval(first), ast.literal_eval(second)
            r = compare(first, second)
            # print(i + 1, first, second, r)
            if r < 0:
                res += i + 1
        return res
    else:
        lst = [ast.literal_eval(it) for it in data]
        a, b = [[2]], [[6]]
        lst.append(a)
        lst.append(b)
        lst.sort(key=functools.cmp_to_key(compare))
        for i, it in enumerate(lst):
            if it == a:
                n1 = i + 1
            if it == b:
                n2 = i + 1
        return n1 * n2


def problem14(data, second):
    _data = split_data('''498,4 -> 498,6 -> 496,6
503,4 -> 502,4 -> 502,9 -> 494,9''')
    fld = np.full((500, 1000), '.')

    def fullslice(start, stop):
        if start < stop: return slice(start, stop + 1)
        return slice(stop, start + 1)

    def parseintpair(p):
        return tuple(int(x) for x in p.split(','))

    max_i = 0
    for line in data:
        for p1, p2 in pairwise(line.split('->')):
            p1, p2 = map(parseintpair, [p1, p2])
            max_i = max(p1[1], p2[1], max_i)
            if p1[0] == p2[0]:
                fld[fullslice(p1[1], p2[1]), p1[0]] = '#'
            else:
                assert p1[1] == p2[1]
                fld[p2[1], fullslice(p1[0], p2[0])] = '#'

    # print('\n'.join(''.join(c for c in s) for s in fld[0:20, 490:510]))

    sys.setrecursionlimit(10_000)

    def pour(row, col):
        try:
            if fld[row, col] != '.':
                return False
        except IndexError:
            return True

        fld[row, col] = '@'
        for i in [0, -1, 1]:
            x = pour(row + 1, col + i)
            if x:
                return x
        return False

    if second:
        fld[max_i + 2, 0 : 1000] = '#'

    pour(0, 500)
    return np.count_nonzero(fld == '@') - (0 if second else 500)



##########

def problem(data, second):
    data = split_data(''' ''')
    if second: return
    return None


##########

if __name__ == '__main__':
    print('Hello')
    # solve_all()
    # solve_latest(22)
    solve_latest()
