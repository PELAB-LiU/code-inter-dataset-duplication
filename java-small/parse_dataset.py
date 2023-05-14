import glob
import json

import javalang
import javalang as jl
from tqdm import tqdm


def __get_start_end_for_node(node_to_find):
    start = None
    end = None
    for path, node in tree:
        if start is not None and node_to_find not in path:
            end = node.position
            return start, end
        if start is None and node == node_to_find:
            start = node.position
    return start, end


def __get_string(start, end, raw_content):
    if start is None:
        return ""

    # positions are all offset by 1. e.g. first line -> lines[0], start.line = 1
    end_pos = None

    if end is not None:
        end_pos = end.line - 1

    lines = raw_content.splitlines(True)
    string = "".join(lines[start.line:end_pos])
    string = lines[start.line - 1] + string

    # When the method is the last one, it will contain a additional brace
    if end is None:
        left = string.count("{")
        right = string.count("}")
        if right - left == 1:
            p = string.rfind("}")
            string = string[:p]

    return string


snippets = []
error_files = 0
for f in tqdm(glob.glob('java-small/**/*.java', recursive=True)):
    try:
        with open(f, 'r') as file:
            contents = file.read()

        tree = jl.parse.parse(contents)
        for _, node in tree.filter(jl.tree.MethodDeclaration):
            try:
                start, end = __get_start_end_for_node(node)
                snippets.append(__get_string(start, end, contents))
            except:
                print('Error parsing a method in: ' + f)
                continue
        for _, node in tree.filter(jl.tree.ConstructorDeclaration):
            try:
                start, end = __get_start_end_for_node(node)
                snippets.append(__get_string(start, end, contents))
            except:
                print('Error parsing a constructor in: ' + f)
                continue
    except:
        print('Error parsing file: ' + f)
        error_files += 1
        continue

print(f'Error files: {error_files}')
# TODO: use utils functions
with open('data.jsonl', 'w') as file:
    for i, snippet in enumerate(snippets):
        tokens = javalang.tokenizer.tokenize(snippet)
        tokens = [str(t.value) for t in tokens]
        json_string = json.dumps({"id_within_dataset": i,
                                  "snippet": snippet,
                                  "tokens": tokens})
        file.write(json_string + '\n')
