-- Up
CREATE TABLE sample (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    content TEXT NOT NULL,
    processed BOOLEAN NOT NULL DEFAULT false
);

-- Down
DROP TABLE sample;
