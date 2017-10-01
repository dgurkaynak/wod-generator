const rp = require('request-promise');
const { init: initDatabase } = require('../preprocessor/db');
const wod = require('../preprocessor/wod');


async function main() {
    console.log('Opening database');
    await initDatabase();

    const allWods = [];
    const MAX_PAGE = 194;

    for (let page = 1; page <= MAX_PAGE; page++) {
        const uri = `https://www.crossfit.com/workout/?page=${page}`;
        console.log(`Fetching ${uri}`);
        const wods = await fetch(uri);
        allWods.push(...wods);
        console.log(`Got ${wods.length} wods, insering...`);

        for (let content of wods) {
            await wod.create({
                content: content.trim(),
                processed: false
            });
        }
    }

    console.log(`Done, fetched total ${allWods.length} wods`);
}


async function fetch(uri) {
    const result = await rp({
        uri,
        headers: {
            'Accept': 'application/json, text/javascript, */*; q=0.01'
        },
        json: true
    });

    return result.wods.map(item => item.wodRaw);
}


main().catch((err) => {
    console.error(err);
    process.exit(1);
})
