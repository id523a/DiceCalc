'''
dice_calc
Copyright (c) 2021 Edward Giles

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
'''

from enum import Enum
import math
from re_lexer import Lexer, LexError, LexerToken
from itertools import chain as iter_chain
from secrets import randbelow as secure_randint
from collections import namedtuple

import numpy as np

class ParseError(Exception):
    pass

class TokenType(Enum):
    END = 0
    DIE_NUMBER = 1
    NUMBER = 2
    PLUS = 3
    MINUS = 4
    MULTIPLY = 5
    DIVIDE = 6
    MULTIPLY_NEG = 7
    DIVIDE_NEG = 8
    EXIT_COMMAND = 9
    STAT_COMMAND = 10
    ROLL_COMMAND = 11
    COMMA = 12
    AC_COMMAND = 13
    DC_COMMAND = 14

input_lexer = Lexer([
        (r"d\s*(?P<sides>[1-9][0-9]*)", TokenType.DIE_NUMBER, lambda m: int(m.group('sides'))),
        (r"[0-9]+", TokenType.NUMBER, lambda m: int(m.group())),
        (r"\+", TokenType.PLUS),
        (r"-", TokenType.MINUS),
        (r"\*", TokenType.MULTIPLY),
        (r"/", TokenType.DIVIDE),
        (r"stat", TokenType.STAT_COMMAND),
        (r"roll", TokenType.ROLL_COMMAND),
        (r",", TokenType.COMMA),
        (r"ac", TokenType.AC_COMMAND),
        (r"check|dc", TokenType.DC_COMMAND),
        (r"exit|^x", TokenType.EXIT_COMMAND),
        (r"\s+", None)
])

operator_inverse = {
    TokenType.PLUS: TokenType.MINUS,
    TokenType.MINUS: TokenType.PLUS,
    TokenType.MULTIPLY: TokenType.MULTIPLY_NEG,
    TokenType.MULTIPLY_NEG: TokenType.MULTIPLY,
    TokenType.DIVIDE: TokenType.DIVIDE_NEG,
    TokenType.DIVIDE_NEG: TokenType.DIVIDE
}

operator_set = operator_inverse.keys()

def parse_token_list(tokens):
    result = []
    operator = TokenType.PLUS
    last_number = 0
    last_token = TokenType.PLUS
    for tok in iter_chain(tokens, [LexerToken(TokenType.END, None)]):
        if last_token in operator_set:
            if tok.token_type is TokenType.PLUS:
                pass
            elif tok.token_type is TokenType.MINUS:
                operator = operator_inverse[operator]
            elif tok.token_type is TokenType.NUMBER:
                last_number = tok.value
            elif tok.token_type is TokenType.DIE_NUMBER:
                result.append((operator, 1, tok.value))
            else:
                raise ParseError(f"Parse error: Unexpected {tok.token_type.name}.")
        elif last_token is TokenType.NUMBER:
            if tok.token_type in operator_set:
                result.append((operator, last_number))
                operator = tok.token_type
            elif tok.token_type is TokenType.DIE_NUMBER:
                result.append((operator, last_number, tok.value))
            elif tok.token_type is TokenType.END:
                result.append((operator, last_number))
                break
            else:
                raise ParseError(f"Parse error: Unexpected {tok.token_type.name}.")
        elif last_token is TokenType.DIE_NUMBER:
            if tok.token_type in operator_set:
                operator = tok.token_type
            elif tok.token_type is TokenType.END:
                break
            else:
                raise ParseError(f"Parse error: Unexpected {tok.token_type.name}.")
        last_token = tok.token_type 
    return result

class Distribution:
    quantile_epsilon = 1e-10
    def __init__(self, origin, vector = None):
        self.cdf_cache = None
        if isinstance(origin, Distribution):
            self.origin = origin.origin
            self.vector = origin.vector
        elif isinstance(origin, int):
            self.origin = origin
            if vector is None:
                vector = np.ones(1)
            if not isinstance(vector, np.ndarray):
                raise TypeError("Second argument (if present) must be a NumPy array.")
            self.vector = vector
        else:
            raise TypeError()

    def normalize(self):
        factor = self.vector.sum()
        if factor != 1.0:
            self.cdf_cache = None
        self.vector /= factor

    def normalized(self):
        return self.vector / self.vector.sum()

    def lower_bound(self):
        return self.origin

    def upper_bound(self):
        return self.origin + self.vector.size - 1

    def mean_without_origin(self):
        self.normalize()
        return np.sum(self.vector * range(self.vector.size))

    def mean(self):
        return self.mean_without_origin() + self.origin

    def variance(self):
        self_mean = self.mean_without_origin()
        return np.sum(self.vector * np.square(np.subtract(range(self.vector.size), self_mean)))

    def ensure_cdf(self):
        if self.cdf_cache is None:
            self.cdf_cache = np.cumsum(self.vector)
            self.cdf_cache /= self.cdf_cache[-1]

    def quantile(self, cutoff):
        self.ensure_cdf()
        a = np.searchsorted(self.cdf_cache, cutoff - Distribution.quantile_epsilon)
        b = np.searchsorted(self.cdf_cache, cutoff + Distribution.quantile_epsilon)
        return self.origin + (a + b) / 2

    def median(self):
        return self.quantile(0.5)

    def mode(self):
        return np.argmax(self.vector) + self.origin

    def lookup_cdf(self, int_threshold):
        self.ensure_cdf()
        shift_threshold = int_threshold - self.origin
        if shift_threshold < 0:
            return 0.0
        elif shift_threshold >= self.cdf_cache.size:
            return 1.0
        else:
            return self.cdf_cache[shift_threshold]
        
    def prob_less_or_equal(self, threshold):
        int_threshold = int(threshold)
        if threshold < int_threshold:
            int_threshold -= 1
        return self.lookup_cdf(int_threshold)

    def prob_greater_than(self, threshold):
        return 1.0 - self.prob_less_or_equal(threshold)

    def prob_less_than(self, threshold):
        int_threshold = int(threshold)
        if threshold <= int_threshold:
            int_threshold -= 1
        return self.lookup_cdf(int_threshold)

    def prob_greater_or_equal(self, threshold):
        return 1.0 - self.prob_less_than(threshold)
    
    def __getitem__(self, idx):
        if not isinstance(idx, int):
            raise TypeError("Index must be an integer.")
        shift_idx = idx - self.origin
        if shift_idx < 0 or shift_idx >= self.vector.size:
            return 0.0
        else:
            return self.vector[shift_idx]

    def __setitem__(self, idx, data):
        if not isinstance(idx, int):
            raise TypeError("Index must be an integer.")
        self.cdf_cache = None
        shift_idx = idx - self.origin
        if shift_idx < 0:
            self.vector = np.pad(self.vector, (-shift_idx,0))
            self.origin += shift_idx
            shift_idx = 0
        elif shift_idx >= self.vector.size:
            self.vector.resize(shift_idx + 1)
        self.vector[shift_idx] = data
            
    def __repr__(self):
        return f"Distribution({self.origin!r},{self.vector!r})"

    def __add__(self, x):
        if isinstance(x, Distribution):
            conv = np.convolve(self.vector, x.vector)
            return Distribution(self.origin + x.origin, conv / conv.sum())
        elif isinstance(x, int):
            return Distribution(self.origin + x, self.vector)
        else:
            raise TypeError()

    def __neg__(self):
        return Distribution(-self.origin - self.vector.size + 1, np.flip(self.vector))

    def __sub__(self, x):
        return self + (-x)

    def __mul__(self, other):
        other = Distribution(other)
        self_lb = self.origin
        self_ub = self.origin + self.vector.size - 1
        other_lb = other.origin
        other_ub = other.origin + other.vector.size - 1
        product_ll = self_lb * other_lb
        product_lu = self_lb * other_ub
        product_ul = self_ub * other_lb
        product_uu = self_ub * other_ub
        product_lb = min(product_ll, product_lu, product_ul, product_uu)
        product_ub = max(product_ll, product_lu, product_ul, product_uu)
        product_vector = np.zeros(product_ub - product_lb + 1)
        for self_idx in range(self_lb, self_ub + 1):
            for other_idx in range(other_lb, other_ub + 1):
                product_vector[self_idx * other_idx - product_lb] += self.vector[self_idx - self_lb] * other.vector[other_idx - other_lb]
        return Distribution(product_lb, product_vector / product_vector.sum())

    def __floordiv__(self, other):
        other = Distribution(other)
        self_lb = self.origin
        self_ub = self.origin + self.vector.size - 1
        other_lb = other.origin
        other_ub = other.origin + other.vector.size - 1
        result = None
        for self_idx in range(self_lb, self_ub + 1):
            for other_idx in range(other_lb, other_ub + 1):
                prob = self.vector[self_idx - self_lb] * other.vector[other_idx - other_lb]
                if prob != 0:
                    value = self_idx // other_idx
                    if result is None:
                        result = Distribution(value, np.array([prob]))
                    else:
                        result[value] += prob
        result.normalize()
        return result

def roll_dice(count, sides):
    if count > 30:
        print(f"Rolling {count} d{sides}")
        total = 0
        for i in range(count):
            total += 1 + secure_randint(sides)
        return total
    else:
        rolls = [1 + secure_randint(sides) for i in range(count)]
        print(f"Rolling {count} d{sides}: " + ', '.join(str(roll) for roll in rolls))
        return sum(rolls)

def roll_action(tokens):
    result = 0
    for op_tuple in parse_token_list(tokens):
            operation = op_tuple[0]
            value = op_tuple[1]
            if len(op_tuple) == 3:
                value = roll_dice(op_tuple[1], op_tuple[2])
            if operation is TokenType.PLUS:
                result += value
            elif operation is TokenType.MINUS:
                result -= value
            elif operation is TokenType.MULTIPLY:
                result *= value
            elif operation is TokenType.DIVIDE:
                result //= value
            elif operation is TokenType.MULTIPLY_NEG:
                result *= -value
            elif operation is TokenType.DIVIDE_NEG:
                result = (-result) // value
    print(f"Result: {result}\n")

def dice_distribution(count, sides):
    if count < 0:
        raise ValueError()
    elif count == 0:
        return Distribution(0)
    elif count == 1:
        return Distribution(1, np.ones(sides) / sides)
    else:
        halfdice = dice_distribution(count // 2, sides)
        return halfdice + halfdice + dice_distribution(count % 2, sides)

def distribution_from_tokens(tokens):
    result = Distribution(0)
    for op_tuple in parse_token_list(tokens):
        operation = op_tuple[0]
        value = op_tuple[1]
        if len(op_tuple) == 3:
            value = dice_distribution(op_tuple[1], op_tuple[2])
        if operation is TokenType.PLUS:
            result += value
        elif operation is TokenType.MINUS:
            result -= value
        elif operation is TokenType.MULTIPLY:
            result *= value
        elif operation is TokenType.DIVIDE:
            result //= value
        elif operation is TokenType.MULTIPLY_NEG:
            result *= -value
        elif operation is TokenType.DIVIDE_NEG:
            result = (-result) // value
    return result

def stat_action(tokens):
    distr = distribution_from_tokens(tokens)
    print(f'''   Mean:{distr.mean():>10.3f}
St. dev:{math.sqrt(distr.variance()):>10.3f}
 
    Min:{distr.lower_bound():>8.1f}
     1%:{distr.quantile(0.01):>8.1f}
    25%:{distr.quantile(0.25):>8.1f}
    50%:{distr.quantile(0.50):>8.1f}
    75%:{distr.quantile(0.75):>8.1f}
    99%:{distr.quantile(0.99):>8.1f}
    Max:{distr.upper_bound():>8.1f}
''')

def check_action(tokens, is_attack_roll):
    if len(tokens) < 1 or not tokens[0].token_type is TokenType.NUMBER:
        raise ParseError("Expected NUMBER after check command.")
    threshold_val = tokens[0].value
    if len(tokens) >= 2 and tokens[1].token_type is TokenType.COMMA:
        tokens = tokens[2:]
    else:
        tokens = tokens[1:]
    distr = distribution_from_tokens(tokens)
    prob_fail = distr.prob_less_than(threshold_val)
    if is_attack_roll:
        prob_crit = distr[distr.upper_bound()]
        prob_fail = max(prob_fail, distr[distr.lower_bound()])
        prob_fail = min(prob_fail, 1 - prob_crit)
        print(f'''   Hit:{100*(1-prob_fail):>8.2f}%
  Crit:{100*prob_crit:>8.2f}%
  Miss:{100*prob_fail:>8.2f}%
''')
    else:
        print(f'''  Pass:{100*(1-prob_fail):>8.2f}%
  Fail:{100*prob_fail:>8.2f}%
''')

special_actions = {
    TokenType.STAT_COMMAND: stat_action,
    TokenType.ROLL_COMMAND: roll_action,
    TokenType.AC_COMMAND: lambda tok: check_action(tok, is_attack_roll=True),
    TokenType.DC_COMMAND: lambda tok: check_action(tok, is_attack_roll=False)
}
prev_input_text = 'exit'
while True:
    try:
        input_text = input("ROLL: ").lower().strip()
        if input_text == '':
            input_text = prev_input_text
        prev_input_text = input_text

        tokens = list(input_lexer.tokenize(input_text))
        if tokens[0].token_type is TokenType.EXIT_COMMAND:
            break

        current_action = roll_action
        if tokens[0].token_type in special_actions:
            current_action = special_actions[tokens[0].token_type]
            tokens.pop(0)

        current_action(tokens)
    except (ParseError, LexError) as e:
        print(e)
    except ZeroDivisionError:
        print("Error: division by zero")
