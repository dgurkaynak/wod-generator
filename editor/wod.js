const db = require('sqlite');


function get(id) {
    return db.get('SELECT * FROM wod WHERE id = ?', id);
}


function getAll() {
    return db.all('SELECT * FROM wod;');
}


function create(data) {
    return db.run(
        `INSERT INTO
            wod (content, processed)
        VALUES
            ($content, $processed)`,
        {
            $content: data.content,
            $processed: data.processed
        }
    );
}


function update(data) {
    return db.run(
        `UPDATE
            wod
        SET
            content = $content,
            processed = $processed
        WHERE
            id = $id`,
        {
            $id: data.id,
            $content: data.content,
            $processed: data.processed
        }
    );
}


function remove(id) {
    return db.run('DELETE FROM wod WHERE id = ?', id);
}


function getRandomUnprocessed() {
    return db.get('SELECT * FROM wod WHERE processed = 0 ORDER BY RANDOM() LIMIT 1;');
}


async function stats() {
    const results = await db.all('SELECT processed, count(*) as count FROM wod GROUP BY processed;');
    const unprocessedRow = results.find(row => row.processed == 0);
    const processedRow = results.find(row => row.processed == 1);
    const unprocessed = unprocessedRow ? unprocessedRow.count : 0;
    const processed = processedRow ? processedRow.count : 0;
    const total = unprocessed + processed
    return {
        unprocessed,
        processed,
        total,
        ratio: {
            unprocessed: (unprocessed / total * 100).toFixed(2),
            processed: (processed / total * 100).toFixed(2)
        }
    };
}


module.exports = {
    get,
    getAll,
    create,
    update,
    remove,
    getRandomUnprocessed,
    stats
};
