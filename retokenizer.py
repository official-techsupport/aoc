import re
import string
import inspect
from dataclasses import dataclass, make_dataclass
from typing import Callable, List, Tuple, Union, Match, Pattern, Iterable


@dataclass
class SubHandler:
    offset: int
    count: int
    handler: Callable


def get_return_value_annotation(f):
    try:
        sig = inspect.signature(f)
    except ValueError:
        return f
    return sig.return_annotation


def compose_alternatives(lst: List[Tuple[Union[str, Pattern], Callable]], flags=0) \
        -> Tuple[Pattern, Callable[[Match], Tuple]]:
    '''Compose several alternatives into a regex. Only one is matched,
    the corresponding handler is called with all explicitly specified groups as *args.
    If there are no explicitly specified capturing groups, the whole match is used.
    '''
    handlers = []
    pieces = []
    offset = 1
    for rx, handler in lst:
        if isinstance(rx, str):
            # compile first to make sure the regex is syntactically correct without the wrapping
            # group (or else consider: '?:abcd', 'ab)(cd' ).
            rx = re.compile(rx, flags)
        count = rx.groups

        pattern = rx.pattern
        # always wrap the pattern in an extra group to detect match (else consider '(a)?bc(d)')
        pattern = '(' + pattern + ')'
        count += 1

        handlers.append(SubHandler(offset, count, handler))
        offset += count
        pieces.append(pattern)

    def combined_handler(match: Match):
        for h in handlers:
            if match.start(h.offset) >= 0:
                if h.count == 1:
                    # no user-defined matching groups
                    args : Tuple = (match[h.offset],)
                else:
                    args = tuple(match[i] for i in range(h.offset + 1, h.offset + h.count))
                return h.handler(*args)

    s = '|'.join(pieces)
    return re.compile(s), combined_handler


def parse_int_or_id(s : str):
    try:
        return int(s)
    except ValueError:
        return s


class ReTokenizer(object):
    r'''Simple regex-based matching solution

    >>> rt = ReTokenizer()
    >>> rt.add_tuple('to tuple {int}')
    >>> rt.fullmatch('to tuple 42')
    (42,)
    >>> rt.add_tuple('with handler {int} {int}', lambda x, y: x + y)
    >>> rt.fullmatch('with handler 13 17')
    30
    >>> rt.add_tuple(r'embedded rx, whole match \w+ zzz')
    >>> rt.fullmatch('embedded rx, whole match hey zzz')
    ('embedded rx, whole match hey zzz',)
    >>> rt.add_tuple('custom rx in group {([0-9]+)}')
    >>> rt.fullmatch('custom rx in group 13')
    ('13',)

    >>> @rt.add_dataclass('dataclass {} {} {} {}', frozen=False)
    ... class Thing:
    ...     a: int
    ...     b: str
    ...     c: 'int_or_id'
    ...     d: '([x]+)'
    >>> rt.fullmatch('dataclass 1 id 3 xx')
    Thing(a=1, b='id', c=3, d='xx')
    '''

    def __init__(self, patterns=[], flags=0):
        self.format_specs = {
            'int': (r'[+-]?\d+', int),
            'id': (r'[A-Za-z_][A-Za-z_0-9-]*', str),
            'int_or_id': (r'[A-Za-z0-9_-]+', parse_int_or_id),
        }
        # alternative notation for dataclasses
        self.format_specs[int] = self.format_specs['int']
        self.format_specs[str] = self.format_specs['id']

        self.flags = flags
        self.alternatives = []
        self.last_compiled = len(self.alternatives)
        if isinstance(patterns, str):
            self.add_tuple(patterns, None)
        else:
            for pattern, user_handler in patterns:
                self.add_tuple(pattern, user_handler)


    def _lazy_init(self):
        if self.last_compiled == len(self.alternatives):
            return
        self.rx, self.handler = compose_alternatives(self.alternatives, self.flags)
        self.last_compiled = len(self.alternatives)


    def get_pattern(self):
        'For debugging purposes'
        self._lazy_init()
        return self.rx.pattern


    def _parse_template(self, s: str, with_names=False):
        fmt = string.Formatter().parse(s)
        rxs = []
        names = []
        handlers = []
        for literal_text, field_name, format_spec, conversion in fmt:
            rxs.append(literal_text)
            if not (field_name or format_spec or conversion):
                # last literal_text
                continue
            assert not conversion
            if with_names:
                assert field_name
                assert format_spec
                names.append(field_name)
            else:
                if not format_spec:
                    assert field_name
                    format_spec = field_name
                    field_name = None
                assert not field_name
            if format_spec.startswith('('):
                rx, handler = format_spec, str
            else:
                rx, handler = self.format_specs[format_spec]
            if not rx.startswith('('):
                rx = '(' + rx + ')'
            rxs.append(rx)
            handlers.append(handler)

        res_rx = re.compile(''.join(rxs), self.flags)
        assert res_rx.groups == len(handlers), 'Don\'t use capturing groups in template'
        return res_rx, names, handlers


    def add_tuple(self, fmt: str, user_handler: Callable=None):
        '''Add a pattern that calls a handler with matched arguments (or the whole string),
        or returns them as a tuple.'''
        rx, _, handlers = self._parse_template(fmt)
        def combined_handler(*args):
            if not len(handlers):
                # no capturing groups
                assert len(args) == 1
            else:
                assert len(args) == len(handlers)
                args = tuple(h(arg) for h, arg in zip(handlers, args))
            if user_handler is not None:
                return user_handler(*args)
            return args
        self.alternatives.append((rx, combined_handler))


    def add_dataclass_old(self, fmt: str, name: str = None, frozen=True):
        rx, names, handlers = self._parse_template(fmt, True)
        assert len(handlers), 'Use add_tuple'
        if name is None:
            name = 'Dataclass'
        cls = make_dataclass(name, names, frozen=frozen)
        def combined_handler(*args):
            assert len(args) == len(handlers)
            args = tuple(h(arg) if arg is not None else arg for h, arg in zip(handlers, args))
            return cls(*args)
        self.alternatives.append((rx, combined_handler))
        return cls


    def add_dataclass(self, fmt: str, /, *, frozen=True, **kwargs):
        fmt = string.Formatter().parse(fmt)
        rxs = []
        positions = []
        for literal_text, field_name, format_spec, conversion in fmt:
            rxs.append(literal_text)
            assert not field_name
            assert not format_spec
            assert not conversion
            if field_name is None: # as opposed to ''
                # last literal_text
                continue
            # reserve place
            positions.append(len(rxs))
            rxs.append(None)

        def wrap(cls):
            handlers = []
            annotations = getattr(cls, '__annotations__', {})
            assert len(positions) == len(annotations)
            for pos, (var, typespec) in zip(positions, annotations.items()):
                rx_str, handler = self.format_specs.get(typespec, (None, None))
                if rx_str is None:
                    # currently only support regex conversions, in the future maybe use tuples
                    assert isinstance(typespec, str)
                    assert typespec.startswith('(')
                    rx_str, handler = typespec, str

                if not rx_str.startswith('('):
                    rx_str = '(' + rx_str + ')'
                rxs[pos] = rx_str
                handlers.append(handler)

                # patch the type
                annotations[var] = get_return_value_annotation(handler)

            rx = re.compile(''.join(rxs), self.flags)
            assert rx.groups == len(handlers), 'Don\'t use capturing groups in template'

            cls = dataclass(cls, frozen=frozen, **kwargs)

            def combined_handler(*args):
                if len(handlers) == 0:
                    assert len(args) == 1
                    return cls()

                assert len(args) == len(handlers)
                args = tuple(h(arg) if arg is not None else arg for h, arg in zip(handlers, args))
                return cls(*args)

            self.alternatives.append((rx, combined_handler))
            return cls

        return wrap


    def fullmatch(self, s: str):
        'Tries to match the whole string'
        self._lazy_init()
        m = self.rx.fullmatch(s)
        assert m is not None, 'Failed to match {!r}'.format(s)
        return self.handler(m)


    def match_all(self, lst: Iterable[str]):
        'fullmatch on every item'
        return [self.fullmatch(s) for s in lst]


    def find(self, s: str):
        'Find a single match in the string'
        self._lazy_init()
        m = self.rx.search(s)
        assert m
        return self.handler(m)


    # be consistent with inconsistent re naming
    search = find


    def find_all(self, s: str):
        'Call handlers on all matches in the string, return a list of results'
        self._lazy_init()
        res = []
        for m in self.rx.finditer(s):
            mm = self.handler(m)
            res.append(mm)
        return res


    def parse(self, s: str):
        'Like find_all(), but checks that there\'s nothing but whitespace besides the matches'
        self._lazy_init()
        res = []
        prev_idx = 0
        def check_space(new_idx):
            space = s[prev_idx:new_idx]
            assert not space or space.isspace(), f'Failed to match at {prev_idx}: {space!r}'

        for m in self.rx.finditer(s):
            check_space(m.start())
            prev_idx = m.end()
            mm = self.handler(m)
            res.append(mm)
        return res


if __name__ == "__main__":
    import doctest
    doctest.testmod()
