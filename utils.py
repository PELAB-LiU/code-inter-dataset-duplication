import tokenize
import io
from typing import re

import javalang

FILTER_TOKENS_PYTHON = [tokenize.NEWLINE, tokenize.INDENT, tokenize.DEDENT, tokenize.ENCODING, tokenize.COMMENT,
                        tokenize.NL, tokenize.ENDMARKER]


def remove_comments_and_docstrings_python(source):
    io_obj = io.StringIO(source)
    out = ""
    prev_toktype = tokenize.INDENT
    last_lineno = -1
    last_col = 0
    for tok in tokenize.generate_tokens(io_obj.readline):
        token_type = tok[0]
        token_string = tok[1]
        start_line, start_col = tok[2]
        end_line, end_col = tok[3]
        ltext = tok[4]
        if start_line > last_lineno:
            last_col = 0
        if start_col > last_col:
            out += (" " * (start_col - last_col))
        if token_type == tokenize.COMMENT:
            pass
        elif token_type == tokenize.STRING:
            if prev_toktype != tokenize.INDENT:
                if prev_toktype != tokenize.NEWLINE:
                    if start_col > 0:
                        out += token_string
        else:
            out += token_string
        prev_toktype = token_type
        last_col = end_col
        last_lineno = end_line
    out = '\n'.join(l for l in out.splitlines() if l.strip())
    return out


def remove_comments_and_docstrings_java_js(string):
    """Source: https://stackoverflow.com/questions/2319019/using-regex-to-remove-comments-from-source-files"""
    pattern = r"(\".*?\"|\'.*?\')|(/\*.*?\*/|//[^\r\n]*$)"
    # first group captures quoted strings (double or single)
    # second group captures comments (//single-line or /* multi-line */)
    regex = re.compile(pattern, re.MULTILINE | re.DOTALL)

    def _replacer(match):
        # if the 2nd group (capturing comments) is not None,
        # it means we have captured a non-quoted (real) comment string.
        if match.group(2) is not None:
            return ""  # so we will return empty to remove the comment
        else:  # otherwise, we will return the 1st group
            return match.group(1)  # captured quoted-string

    return regex.sub(_replacer, string)


def get_tokens_from_python_snippet(snippet):
    tokens = tokenize.tokenize(io.BytesIO(snippet.encode('utf-8')).readline)
    result = []
    for token in tokens:
        if token.type not in FILTER_TOKENS_PYTHON:
            result.append(token.string)
    return result


def get_tokens_from_java_snippet(snippet):
    tokens = javalang.tokenizer.tokenize(snippet)
    tokens = [str(t.value) for t in tokens]
    return tokens


def get_tokens_from_snippet(snippet, language):
    if language == 'python':
        snippet_without_comments = remove_comments_and_docstrings_python(snippet)
        return get_tokens_from_python_snippet(snippet_without_comments)
    elif language == 'java':
        snippet_without_comments = remove_comments_and_docstrings_java_js(snippet)
        return get_tokens_from_java_snippet(snippet_without_comments)
    else:
        raise ValueError(f'Language {language} not supported')
