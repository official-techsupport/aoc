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
                elif s == '..':
                    current = current['..']
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
