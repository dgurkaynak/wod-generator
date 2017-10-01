require('dotenv').config();
const Hapi = require('hapi');
const path = require('path');
const Vision = require('vision');
const Handlebars = require('handlebars');
const inert = require('inert');
const { init: initDatabase } = require('./db');
const wod = require('./wod');
const Boom = require('boom');


async function main() {
    await initDatabase();

    const server = create();
    initRoutes(server);
    await start(server);
}


function create() {
    const server = new Hapi.Server();
    server.connection({ port: process.env.PREPROCESSOR_PORT });
    server.register(Vision, (err) => {
        if (err) {
            console.error('Cannot register vision');
        }

        server.views({
            engines: {
                html: Handlebars
            },
            path: __dirname + '/views'
        });
    });
    server.register(inert, (err) => {
        if (err) {
            console.error('Cannot register inert');
        }
    });

    return server;
}


function start(server) {
    return new Promise((resolve, reject) => {
        server.start((err) => {
            if (err) {
                console.error('Could not start server', err);
                return reject(err);
            }

            console.log(`Server running at: ${server.info.uri}`);
            resolve();
        });
    });
}


function initRoutes(server) {
    server.route({
        method: 'GET',
        path: process.env.PREPROCESSOR_BASE_PATH,
        handler: async (request, reply) => {
            const [row, stats] = await Promise.all([
                wod.getRandomUnprocessed(),
                wod.stats()
            ]);
            reply.view('index', { wod: row, stats });
        }
    });

    server.route({
        method: 'POST',
        path: process.env.PREPROCESSOR_BASE_PATH,
        handler: async (request, reply) => {
            if (!request.payload.id) return reply(Boom.badRequest('Required field `id`'));
            if (!request.payload.content) return reply(Boom.badRequest('Required field `content`'));
            const id = parseInt(request.payload.id, 10);
            if (isNaN(id)) return reply(Boom.badRequest('Invalid `id`'));

            try {
                await wod.update({
                    id,
                    content: request.payload.content.trim(),
                    processed: 1
                });
                reply.redirect(process.env.PREPROCESSOR_BASE_PATH);
            } catch (err) {
                console.error('An error occured while POST /', err);
                reply(Boom.badImplementation('Unexpected error, please try again'));
            }
        }
    });

    server.route({
        method: 'DELETE',
        path: process.env.PREPROCESSOR_BASE_PATH,
        handler: async (request, reply) => {
            if (!request.payload.id) return reply(Boom.badRequest('Required field `id`'));
            const id = parseInt(request.payload.id, 10);
            if (isNaN(id)) return reply(Boom.badRequest('Invalid `id`'));

            try {
                await wod.remove(id);
                reply('Deleted');
            } catch (err) {
                console.error('An error occured while DELETE /', err);
                reply(Boom.badImplementation('Unexpected error, please try again'));
            }
        }
    });
}


main().catch((err) => {
    console.error(err);
    process.exit(1);
});
