require('dotenv').config();
const path = require('path');
const fs = require('fs');
const { init: initDatabase } = require('../editor/db');
const wod = require('../editor/wod');


const OUTPUT_FILE = path.join(__dirname, '../data/wods.txt');
const SEPERATOR_CHAR = '\n|\n';


async function main() {
    await initDatabase();
    const wods = await wod.getAll();
    const wodsText = wods.map(wod => wod.content.trim()).join(SEPERATOR_CHAR);
    fs.writeFile(OUTPUT_FILE, wodsText, (err) => {
        if (err) {
            console.error(err);
            process.exit(1);
        }

        console.log(`Written to ${OUTPUT_FILE}`);
    });
}


main().catch((err) => {
    console.error(err);
    process.exit(1);
});
