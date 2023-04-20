CREATE TABLE datasets (
	id TEXT PRIMARY KEY,
	url TEXT NOT NULL,
	tasks TEXT NOT NULL
);

CREATE TABLE snippets (
	id integer primary key autoincrement,
	snippet TEXT NOT NULL,
	nl TEXT,
	dataset TEXT NOT NULL,
	language TEXT NOT NULL,
	tokens TEXT NOT NULL,
	partition TEXT NOT NULL,
	FOREIGN KEY (dataset)
    REFERENCES datasets (id)
);

CREATE TABLE duplicates (
    id integer primary key autoincrement,
    snippet1 INT NOT NULL,
    snippet2 INT NOT NULL,
    FOREIGN KEY (snippet1)
    REFERENCES snippets (id),
    FOREIGN KEY (snippet2)
    REFERENCES snippets (id)
);