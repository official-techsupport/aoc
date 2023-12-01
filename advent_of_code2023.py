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
    rx = re.compile('(?=(' + '|'.join(ds) + '))')
    def get(s: str):
        mm = rx.findall(s)
        def decode(s):
            try:
                return str(ds.index(s))
            except ValueError:
                return str(int(s))
        return int(decode(mm[0]) + decode(mm[-1]))
    return sum(get(s) for s in data)




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