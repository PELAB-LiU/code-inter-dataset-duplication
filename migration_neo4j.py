import argparse
import sqlite3

from neo4j import GraphDatabase

AUTH = ("<Username>", "<Password>")


def add_snippets(driver, conn):
    query_snippet_sql = 'SELECT id, dataset, id_within_dataset, snippet FROM snippets'
    query_cyper_insert = 'CREATE (s:Snippet {id: $id, dataset: $dataset, id_within_dataset: $id_within_dataset, ' \
                         'snippet: $snippet}) '
    # write the results of the sql query to neo4j
    cursor = conn.cursor()
    cursor.execute(query_snippet_sql)
    for id, dataset, id_within_dataset, snippet in cursor:
        with driver.session() as session:
            session.run(query_cyper_insert, id=id, dataset=dataset, id_within_dataset=id_within_dataset,
                        snippet=snippet)


def main(args):
    driver = GraphDatabase.driver(args.neo4j, auth=AUTH)
    conn = sqlite3.connect(args.db)

    add_snippets(driver, conn)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--db', type=str, default='interduplication.db')
    parser.add_argument('--neo4j', type=str, default='neo4j://localhost:7687')
    args = parser.parse_args()
    main(args)
