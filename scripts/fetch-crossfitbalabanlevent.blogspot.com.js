const rp = require('request-promise');
const cheerio = require('cheerio');
const { init: initDatabase } = require('../preprocessor/db');
const wod = require('../preprocessor/wod');


async function main() {
    console.log('Opening database');
    await initDatabase();

    const allWods = [];
    let uri = 'http://crossfitbalabanlevent.blogspot.com.tr/search?max-results=1000';

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

    return {
        wods: $('.post-body').map((i, el) => $(el).text()).toArray(),
        nextURI: $('a.blog-pager-older-link').attr('href')
    };
}


main().catch((err) => {
    console.error(err);
    process.exit(1);
})
