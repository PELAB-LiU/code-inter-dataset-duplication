import string

from dpu_utils.codeutils import get_language_keywords


def filter_tokens(tokens, lang):
    return [t for t in tokens if t not in get_language_keywords(lang)
            and t not in list(string.punctuation)]
