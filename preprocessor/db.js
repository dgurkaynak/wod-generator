require('dotenv').config();
const db = require('sqlite');


module.exports.init = async function() {
    await db.open(process.env.DB_FILE, { Promise });
    await db.migrate({ force: process.env.DB_FORCE_MIGRATIONS ? 'last' : false })
}
