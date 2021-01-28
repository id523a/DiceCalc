'''
re_lexer
Copyright (c) 2020-2021 Edward Giles

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

import re

from collections import namedtuple

def default_lexer_action(match):
    return None

LexerRule = namedtuple("LexerRule", "pattern token_type action", defaults=(default_lexer_action,))
LexerToken = namedtuple("LexerToken", "token_type value")

del default_lexer_action
del namedtuple

class LexError(Exception):
    preview_char_count = 20
    def __init__(self, preview):
        self.preview = preview
        super().__init__("Unexpected characters: " + preview)

class Lexer:
    def __init__(self, lexer_rules, re_flags=0):
        self.rule_lookup = {}
        regex_build = []
        for idx, lex_def in enumerate(lexer_rules):
            if not isinstance(lex_def, LexerRule):
                lex_def = LexerRule(*lex_def)
            rule_name = f"LEX_R{idx}"
            regex_build.append(f"(?P<{rule_name}>{lex_def.pattern})")
            self.rule_lookup[rule_name] = lex_def 
        self.lexer_regex = re.compile("|".join(regex_build), re_flags)

    def tokenize(self, seq, *args, **kwargs):
        if isinstance(seq, str):
            return self.tokenize_str(seq, *args, **kwargs)
        else:
            return self.tokenize_file(seq, *args, **kwargs)

    def raise_lex_error(self, seq, start_point):
        error_end = start_point + LexError.preview_char_count
        ellipsis = ""
        if error_end >= len(seq):
            error_end = len(seq)
        else:
            error_end = error_end - 3
            ellipsis = "..."
        find_newline = seq.find('\n',start_point,error_end)
        if find_newline >= 0:
            error_end = find_newline
            ellipsis = ""
        seq = seq[start_point:error_end]
        raise LexError(repr(seq)[1:-1] + ellipsis)
    
    def tokenize_str(self, seq):
        start_point = 0
        while start_point < len(seq):
            match_obj = self.lexer_regex.match(seq, start_point)
            if match_obj is None:
                self.raise_lex_error(seq, start_point)
            rule = self.rule_lookup[match_obj.lastgroup]
            if rule.token_type is not None:
                yield LexerToken(rule.token_type, rule.action(match_obj))
            start_point = match_obj.end()

    def tokenize_file(self, seq, chunk_size=1024):
        buffer = ''
        eof = False
        start_point = 0
        while (not eof) or start_point < len(buffer):
            # Remove characters at the beginning of the buffer
            if start_point >= chunk_size:
                buffer = buffer[start_point:]
                start_point = 0
            # Fill buffer with characters from file
            while (not eof) and len(buffer) - start_point < chunk_size:
                file_data = seq.read(chunk_size)
                if len(file_data) == 0:
                    eof = True
                else:
                    buffer += file_data
            # Match the next token
            match_obj = self.lexer_regex.match(buffer, start_point)
            if match_obj is None:
                self.raise_lex_error(buffer, start_point)
            rule = self.rule_lookup[match_obj.lastgroup]
            if rule.token_type is not None:
                yield LexerToken(rule.token_type, rule.action(match_obj))
            start_point = match_obj.end()

def re_lexer_main():
    from io import StringIO
    file = StringIO(input("Enter string to tokenize: "))

    x = [
        LexerRule(r"[0-9]+", "NUMBER", lambda m : int(m.group())),
        LexerRule(r"[A-Za-z_][A-Za-z0-9_]*", "WORD"),
        LexerRule(r"\s+", None),
    ]
    for token in Lexer(x).tokenize(file):
        print(token)

if __name__ == "__main__":
    re_lexer_main()

del re_lexer_main

