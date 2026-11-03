// Inline engine.js into app.html -> dist/flyover-locator.html (the single file that gets published).
const fs = require('fs');
const path = require('path');
const engine = fs.readFileSync(path.join(__dirname, 'engine.js'), 'utf8');
if (/<\/script/i.test(engine)) throw new Error('engine.js must not contain </script>');
const page = fs.readFileSync(path.join(__dirname, 'app.html'), 'utf8');
if (!page.includes('/*__ENGINE__*/')) throw new Error('placeholder missing in app.html');
fs.mkdirSync(path.join(__dirname, 'dist'), { recursive: true });
const out = path.join(__dirname, 'dist', 'flyover-locator.html');
fs.writeFileSync(out, page.replace('/*__ENGINE__*/', () => engine));
console.log(`wrote ${out} (${(fs.statSync(out).size / 1024).toFixed(0)} KB)`);
// local preview only: the published page gets this skeleton from the artifact host
const reset = ':root{color-scheme:light}body{margin:0;font:14px system-ui}img{max-width:100%}[hidden]{display:none!important}';
fs.writeFileSync(path.join(__dirname, 'dist', 'preview.html'), `<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover"><style>${reset}</style></head><body>` + fs.readFileSync(out, 'utf8') + '</body></html>');
