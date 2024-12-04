#!/usr/bin/env python3

from pathlib import Path
from typing import Sequence, Tuple, Callable, List, Union, Optional, Any, Set, Iterable, Dict
from collections import Counter, defaultdict, deque, namedtuple
import collections.abc
from enum import Enum
from dataclasses import dataclass, make_dataclass, field as dataclass_field, asdict as dataclass_asdict
from pprint import pprint, pformat
from heapq import heappush, heappop, heapify
from contextlib import contextmanager
import copy, math, cmath, random, operator
import itertools, re, functools, shutil, errno, sys, time, hashlib, json, ast, string
from itertools import pairwise
import requests
import numpy as np
import networkx
# from blist import blist

# line_profiler support
try:
    profile
except:
    def profile(f):
        return f

quiet = True

from retokenizer import ReTokenizer


def removeprefix(prefix, s):
    res = s.removeprefix(prefix)
    assert res != s, f'removeprefix({prefix!r}, {s!r})'
    return res


def random_shuffle(lst):
    return random.sample(lst, len(lst))


def npv2(x, y):
    return np.array([x, y])


def pad_map_with_border(mm, sentinel=None):
    '''Pad a rectangular map with a border made of `None`s or provided sentinel values'''
    width = len(mm[0])
    assert all(width == len(row) for row in mm)
    res = []
    res.append([sentinel for _ in range(width + 2)])
    for row in mm:
        rr = [sentinel]
        rr.extend(row)
        rr.append(sentinel)
        res.append(rr)
    res.append([sentinel for _ in range(width + 2)])
    return res


def grouper(iterable, n, fillvalue=None):
    "Collect data into fixed-length chunks or blocks"
    # grouper('ABCDEFG', 3, 'x') --> ABC DEF Gxx"
    args = [iter(iterable)] * n
    return itertools.zip_longest(*args, fillvalue=fillvalue)


def product_up_to(n: int, k: int):
    '''yield k-tuples consisting of nonnegative numbers summing up to n. They are actually lists, plz don't modify

    >>> for it in product_up_to(2, 3): print(it)
    [2, 0, 0]
    [1, 1, 0]
    [0, 2, 0]
    [1, 0, 1]
    [0, 1, 1]
    [0, 0, 2]
    '''
    indices = [0] * k
    def rec(rk, remaining):
        if rk == 0:
            indices[0] = remaining
            yield indices
        else:
            for i in range(remaining + 1):
                indices[rk] = i
                yield from rec(rk - 1, remaining - i)
    yield from rec(k - 1, n)


def log_seq(seq):
    for n, it in enumerate(seq):
        print(n, it)
        yield it


def np_2d_aabbox(arr, axis=0):
    min_x, min_y = np.min(arr, 0)
    max_x, max_y = np.max(arr, 0)
    w = max_x - min_x + 1
    h = max_y - min_y + 1
    return min_x, w, min_y, h


def complex_2d_aabbox(arr: Iterable[complex]):
    min_x = min(c.real for c in arr)
    min_y = min(c.imag for c in arr)
    max_x = max(c.real for c in arr)
    max_y = max(c.imag for c in arr)
    w = max_x - min_x + 1
    h = max_y - min_y + 1
    return tuple(map(int, (min_x, w, min_y, h)))


def c2v2(c: complex):
    return int(c.imag), int(c.real)


def _life(field):
    nb = sum(np.roll(field, (i - 1, j - 1), axis=(0, 1)) for i in range(3) for j in range(3))
    return (field & (nb == 4)) | (nb == 3)


def np_add_8neighbors(world, destination=None, add_self=0):
    assignments = getattr(np_add_8neighbors, 'assignments', None)
    if not assignments:
        assignments = np_add_8neighbors.assignments = []
        axis = [slice(None), slice(1, None), slice(0, -1)]
        for x in (-1, 0, 1):
            for y in (-1, 0, 1):
                if x == 0 and y == 0:
                    continue
                assignments.append(((axis[x], axis[y]), (axis[-x], axis[-y])))

    if add_self:
        if destination is not None:
            np.multiply(world, add_self, out=destination)
        else:
            destination = world * add_self
    else:
        if destination is not None:
            destination[:, :] = 0
        else:
            destination = np.zeros_like(world)


    for fr, to in assignments:
        destination[to] += world[fr]

    return destination


def ndenumerate_slice(a, np_s_):
    '''like np.ndenumerate on a slice of the array

    >>> a = np.arange(4 * 5).reshape(4, 5)
    >>> list(ndenumerate_slice(a, np.s_[1:-1, 1:-2]))
    [((1, 1), 6), ((1, 2), 7), ((2, 1), 11), ((2, 2), 12)]
    '''
    for p in itertools.product(*(range(*slc.indices(a.shape[i])) for i, slc in enumerate(np_s_))):
        yield p, a[p]


def ndarray_from_chargrid(data):
    if len(data[0]) == 1:
        # returned by get_raw_data()
        data = data.strip('\n').split('\n')
    if isinstance(data[0], str):
        data = [list(row) for row in data]
    return np.array(data, dtype='U1')


def grid_to_graph(grid):
    g = networkx.Graph()
    for p, c in np.ndenumerate(grid[:-1, :-1]):
        if c != '#':
            p1 = addv2(p, (0, 1))
            if grid[p1] != '#':
                g.add_edge(p, p1)
            p1 = addv2(p, (1, 0))
            if grid[p1] != '#':
                g.add_edge(p, p1)
    return g


def precompute_shortest_paths(g, locations):
    '''Returns a graph containing only nodes in locations, with
    weighted edges between them corresponding to original shortest paths.
    Original paths are stored in the "path" attribute of edges (direction is arbitrary).

    Graph is assumed to be undirected and unweighted.

    Uses bfs from each location, not super efficient but probably better than
    all_pairs_shortest_paths on problems with tens of interesting locations and 10k total nodes, or
    where there's a lot of disjoint subgraphs'''

    g2 = networkx.Graph()

    for i, source in enumerate(locations):
        paths = networkx.single_source_shortest_path(g, source)
        for j in range(i + 1, len(locations)):
            target = locations[j]
            if (path := paths.get(target)) is not None:
                g2.add_edge(source, target, weight=len(path) - 1, path=path)
    return g2



Vector2 = Tuple[int, int]

# cw starting from north, y points down.
directions4 = ((0, -1), (1, 0), (0, 1), (-1, 0))
directions8 = ((0, -1), (1, -1), (1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1))
# for dx, dy in directions4:
#     nx, ny = x + dx, y + dy


def addv2(v1, v2):
    return v1[0] + v2[0], v1[1] + v2[1]


def mulv2s(v1, sc):
    return v1[0] * sc, v1[1] * sc


def cmulv2(v1, v2):
    'Complex multiplication, vectors are (row, column)'
    return (v1[0] * v2[1] + v1[1] * v2[0], v1[1] * v2[1] - v1[0] * v2[0])


flatten = itertools.chain.from_iterable


def bfs_extract_path(node, visited):
    path = []
    while node is not None:
        path.append(node)
        node = visited[node]
    path.reverse()
    return path


def bfs_full_step(front, visited, goal, neighbors: Callable):
    '''Returns a list of goals reached on this step'''
    sentinel = object()
    front.append(sentinel)
    goals: List[Any] = []
    while True:
        parent_node = front.popleft()
        if parent_node is sentinel:
            return goals
        for node in neighbors(parent_node):
            if node not in visited:
                visited[node] = parent_node
                front.append(node)
                if goal(node):
                    goals.append(node)


def bfs_full(start, goal: Callable, neighbors: Callable):
    '''Returns (goals, visited) or (None, visited) if goal unreachable or unspecified
    Always does the last step in full.
    '''
    visited = {s: None for s in start}
    front = deque(start)
    step = 1
    while front:
        found = bfs_full_step(front, visited, goal, neighbors)
        if found:
            return found, visited
        if not quiet: print(f'BFS: step={step}, visited={len(visited)}')
        step += 1
    return [], visited


def cached_property(f):
    # relies on the fact that instance attributes override non-data descriptors.
    class Wrapper:
        def __get__(self, obj, cls):
            if obj is None:
                return self
            value = f(obj)
            obj.__dict__[f.__name__] = value
            return value
    return Wrapper()


@contextmanager
def timeit_block(what='Something'):
    import timeit
    t = timeit.default_timer()
    yield
    t2 = timeit.default_timer()
    print(f'{what} took {t2 - t:0.3f} seconds')


##############

class AnswerDb:
    def __init__(self, year, caller_globals):
        self.year = year
        self.caller_globals = caller_globals
        self.file = Path(f'input/ac{year}_answerdb.txt')
        self.warnings = []

    def load(self):
        if not self.file.exists():
            return {}
        return json.loads(self.file.read_text())

    def save(self, answers):
        self.file.write_text(json.dumps(answers, sort_keys=True, indent=4), newline='\n')

    def answer(self, id, answer):
        assert answer is not None
        if np.issubdtype(type(answer), np.integer):
            answer = int(answer)
        answers = self.load()
        orig = answers.get(id)
        if orig is not None and answer != orig:
            warning = f'Different answer for {id!r}, was:\n{pformat(orig)}\nnew:\n{pformat(answer)}'
            self.warnings.append(warning)
            print(warning)
        if orig is None or answer != orig:
            answers[id] = answer
            self.save(answers)

    def print_warnings(self):
        for w in self.warnings:
            print(w + '\n')
        del self.warnings[:]

    def fetch_input(self, n):
        file = Path('input') / f'ac{self.year}d{n}_input.txt'
        try:
            data = file.read_text()
            print('Problem read from {!r}'.format(file.as_posix()))
            return data
        except IOError as exc:
            if exc.errno != errno.ENOENT:
                raise

        session_cookie = Path('aoc_session.txt').read_text().strip()
        url = f'http://adventofcode.com/{self.year}/day/{n}/input'.format(n)
        print('Fetching problem from {!r}'.format(url))
        r = requests.get(url, cookies={'session': session_cookie})
        assert r.status_code == 200
        r.raw.decode_content = True
        with file.open('w', newline='\n') as f:
            f.write(r.text)
        print('Problem cached in {!r}'.format(file.as_posix()))
        return r.text

answer_db: AnswerDb

def utils_init(year, caller_globals):
    global answer_db
    answer_db = AnswerDb(year, caller_globals)


def split_data(s):
    data = [s.strip() for s in s.strip().split('\n')]
    if len(data) == 1:
        [data] = data
    return data


def get_problem_number(problem_function_name):
    m = re.match(r'problem(\d+)$', problem_function_name)
    if m:
        return int(m.group(1))


_raw_data: str
def get_raw_data():
    return _raw_data


def solve(problem: Callable, second=False):
    global _raw_data
    n = get_problem_number(problem.__name__)
    assert n is not None
    data = answer_db.fetch_input(n)
    _raw_data = data
    data = split_data(data)
    description = f'{n}_{int(second) + 1}'
    with timeit_block(description):
        res = problem(data, second)
    if res is not None:
        answer_db.answer(description, res)
    return res


def collect_problems():
    res = []
    for k, v in answer_db.caller_globals.items():
        n = get_problem_number(k)
        if n is not None:
            # magic
            v.n = n
            res.append(v)
    return sorted(res, key=lambda f: f.n)


def solve_all(skip={}):
    for problem in collect_problems():
        print(f'### {problem.n}')
        print(solve(problem))
        if problem.n in skip:
            print('Skipping second')
            continue
        print(solve(problem, True))
    answer_db.print_warnings()


def solve_latest(problem=None, second=None):
    problems = collect_problems()
    if problem is not None:
        p = next(p for p in problems if p.n == problem)
    else:
        p = problems[-1]

    if second is not None:
        print(solve(p, second))
    else:
        print(solve(p))
        print(solve(p, True))


if __name__ == '__main__':
    import doctest
    doctest.testmod()

