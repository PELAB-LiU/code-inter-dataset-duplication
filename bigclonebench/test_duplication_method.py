import csv
import sqlite3

import pandas as pd

DB = 'bigclonebench.db'
FILES = ['bigclonebench/train.txt', 'bigclonebench/test.txt', 'bigclonebench/valid.txt']

positive_pairs = set([])
for file in FILES:
    with open(file, 'r') as f:
        csvreader = csv.reader(f, delimiter='\t')
        for row in csvreader:
            if row[2] == '1':
                positive_pairs.add(frozenset({int(row[0]), int(row[1])}))

conn = sqlite3.connect(DB)

query = '''select s.id_within_dataset as i1, s2.id_within_dataset as i2
from duplicates
inner join snippets s on s.id = duplicates.snippet1
inner join snippets s2 on s2.id = duplicates.snippet2'''

df = pd.read_sql_query(query, conn)
predicted_pairs = set([frozenset({row['i1'], row['i2']})
                   for _, row in df.iterrows()])

precision = len(predicted_pairs.intersection(positive_pairs)) / len(predicted_pairs)
recall = len(predicted_pairs.intersection(positive_pairs)) / len(positive_pairs)
print('Precision:', precision)
print('Recall:', recall)
