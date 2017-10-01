const rp = require('request-promise');
const cheerio = require('cheerio');
const { init: initDatabase } = require('../preprocessor/db');
const wod = require('../preprocessor/wod');


async function main() {
    console.log('Opening database');
    await initDatabase();

    const allWods = [];
    let uri = 'http://www.findawod.com/wods';

    while (uri) {
        console.log(`Fetching ${uri}`);
        const {wods, nextURI} = await fetch(uri);
        allWods.push(...wods);
        uri = nextURI;
        console.log(`Got ${wods.length} wods, inserting...`);

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
    const $ = await rp({
        uri,
        transform: body => cheerio.load(body)
    });

    const nextURI = $('.col.s4 a.btn.right').attr('href');

    return {
        wods: $('ul.wod').map((i, el) => $(el).text()).toArray().map(str => str.replace(/[^\S\r\n]{2,}/ig, '')),
        nextURI: nextURI ? `http://www.findawod.com${nextURI}` : null
    };
}


main().catch((err) => {
    console.error(err);
    process.exit(1);
})
