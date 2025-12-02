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
