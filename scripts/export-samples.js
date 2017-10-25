require('dotenv').config();
const path = require('path');
const fs = require('fs');
const { init: initDatabase } = require('../editor/db');
const db = require('sqlite');


async function main() {
    await initDatabase();
    const results = await db.all('SELECT * FROM sample WHERE processed = 1;');
    const samples = results.map(x => x.content);
    console.log(JSON.stringify(samples));
}


main().catch((err) => {
    console.error(err);
    process.exit(1);
});
