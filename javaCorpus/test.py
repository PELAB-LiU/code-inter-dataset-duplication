CONTENT = 'package ch . hsr . geohash ; import java . util . Random ; import ch . mollusca . benchmarking . Before ; ' \
          'import ch . mollusca . benchmarking . Benchmark ; public class GeoHashEncodingBenchmark { private static ' \
          'final int NUMBER_OF_HASHES = 1000000 ; private GeoHash [ ] hashes ; private double [ ] latitudes ; private ' \
          'double [ ] longitudes ; @ Before public void setupBenchmark ( ) { hashes = new GeoHash [ NUMBER_OF_HASHES ' \
          '] ; latitudes = new double [ NUMBER_OF_HASHES ] ; longitudes = new double [ NUMBER_OF_HASHES ] ; Random ' \
          'rand = new Random ( ) ; for ( int i = 0 ; i < NUMBER_OF_HASHES ; i ++ ) { latitudes [ i ] = rand . ' \
          'nextDouble ( ) * 180 - 90 ; longitudes [ i ] = rand . nextDouble ( ) * 360 - 180 ; } } @ Benchmark ( times ' \
          '= 10 ) public void benchmarkGeoHashEncoding ( ) { for ( int i = 0 ; i < NUMBER_OF_HASHES ; i ++ ) { hashes ' \
          '[ i ] = GeoHash . withBitPrecision ( latitudes [ i ] , longitudes [ i ] , 60 ) ; } } } '

import sys

sys.path.append('..')
from utils import get_methods_java

contents = CONTENT.split()
new_contents = []
for content in contents:
    new_contents.append(content)
    if content == ';' or content == '}' or content == '{':
        new_contents.append('\n')
snippets = get_methods_java(' '.join(new_contents))
print(snippets[0])
print(len(snippets))

import re


def extract_java_functions(code):
    pattern = r'(public|private|protected)?\s+(static\s+)?\w+\s+\w+\s*\([^)]*\)\s*{([^}]*)}'
    matches = re.findall(pattern, code, re.DOTALL)
    functions = []

    for match in matches:
        function_declaration = ' '.join(match[:2]) + ' '.join(
            match[2].split())  # Join access modifier, static modifier, and function declaration
        function_body = match[3].strip()  # Remove leading/trailing whitespace from function body
        function_string = f"{function_declaration} {function_body}"
        functions.append(function_string)

    return functions


print(extract_java_functions(CONTENT))
