import operator
from functools import reduce
import sys
from advent_of_code_utils import timeit_block

class Day11Clean:
	class Monkey:
		def play(self, mkys, worry_reduce):
			items, self.items = self.items, []
			for worry in items:
				worry = self.worry_op(worry)
				worry = worry_reduce(worry)
				self.inspect += 1
				mkys[self.test(worry)].items.append(worry)

		def test(self, worry):
			if worry % self.test_mod == 0: return self.test_t
			return self.test_f

	def _load(self, data, qty):
		mkys = [self.Monkey() for i in range(qty)]
		for i, m in enumerate(mkys):
			m.items    = list(map(int, data[i*7+1].split(': ')[1].split(', ')))
			m.op_str   = data[i*7+2].split('Operation: new = ')[1]
			m.worry_op = (lambda op: eval(f'lambda old: {op}'))(m.op_str)
			m.test_mod = int(data[i*7+3].split('Test: divisible by ')[1])
			m.test_t   = int(data[i*7+4].split('If true: throw to monkey ')[1])
			m.test_f   = int(data[i*7+5].split('If false: throw to monkey ')[1])
			m.inspect  = 0
		return mkys

	def _run(self, mkys, rounds, worry_reduce):
		for i in range(rounds):
			for m in mkys:
				m.play(mkys, worry_reduce)
		business = sorted([m.inspect for m in mkys], reverse=True)
		return business[0] * business[1]

	def run_a(self, data):
		mkys = self._load(data, 8)
		return self._run(mkys, 20, lambda x: x // 3)

	def run_b(self, data, iter = 10000):
		mkys = self._load(data, 8)
		worry_modulus = reduce(operator.mul, [m.test_mod for m in mkys])
		return self._run(mkys, iter, lambda x: x % worry_modulus)

with open('input/ac2022d11_input.txt', 'r') as f:
    data = f.read().splitlines()
    with timeit_block():
        print(Day11Clean().run_b(data))
