// Renders the title cards (1920x1080, opaque) and caption bands (1920x1080, transparent outside the band) from spec.json.
//   node cards.mjs <spec.json> <out dir>
import fs from 'fs'
import { chromium } from 'playwright'   // npm install playwright, in this folder
const [specPath, OUT] = process.argv.slice(2)
const spec = JSON.parse(fs.readFileSync(specPath, 'utf8'))
const FONTS = '<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Sans+Condensed:wght@500;600;700&family=IBM+Plex+Mono:wght@500&display=swap">'
const BASE = `*{box-sizing:border-box}html,body{margin:0;width:1920px;height:1080px}body{font-family:'IBM Plex Sans',sans-serif;color:#e8ebf1}`
const esc = s => s.replace(/&/g, '&amp;').replace(/</g, '&lt;')

function cardHtml(c) {
  const lines = (c.lines || []).map(l => `<p class="${l.kind || ''}">${esc(l.text)}</p>`).join('')
  return `<!doctype html><html><head><meta charset="utf-8">${FONTS}<style>${BASE}
  body{background:#12161d;display:grid;place-content:center;padding:0 200px}
  .brand{font:700 150px/1 'IBM Plex Sans Condensed';letter-spacing:.01em;margin:0 0 28px;color:#fff}
  h1{font:600 64px/1.15 'IBM Plex Sans Condensed';margin:0 0 30px;color:#fff}
  p{font-size:38px;line-height:1.35;margin:0 0 22px;color:#c9d0db;max-width:1400px}
  p.big{font:600 52px/1.2 'IBM Plex Sans Condensed';color:#fff}
  p.red{color:#ff7d85;font-weight:600}
  p.source{font-size:24px;color:#8b94a4;margin-top:-10px}
  p.mono{font:500 32px/1.3 'IBM Plex Mono';color:#8db6ec}
  .rule{width:120px;height:6px;background:#c4252f;margin:0 0 34px}
  </style></head><body>${c.brand ? `<div class="brand">${esc(c.brand)}</div>` : ''}${c.title ? `<div class="rule"></div><h1>${esc(c.title)}</h1>` : ''}${lines}</body></html>`
}

function captionHtml(text) {
  return `<!doctype html><html><head><meta charset="utf-8">${FONTS}<style>${BASE}
  body{background:transparent}
  .band{position:absolute;left:160px;right:160px;top:938px;height:122px;display:flex;align-items:center}
  .band p{margin:0;font:500 34px/1.3 'IBM Plex Sans';color:#f3f5f8}
  .band::before{content:'';flex:0 0 6px;align-self:stretch;margin:14px 26px 14px 0;background:#c4252f}
  </style></head><body><div class="band"><p>${esc(text)}</p></div></body></html>`
}

const browser = await chromium.launch({ channel: 'chrome' })
const page = await browser.newPage({ viewport: { width: 1920, height: 1080 } })
for (const c of spec.cards) {
  await page.setContent(cardHtml(c), { waitUntil: 'networkidle' })
  await page.evaluate(() => document.fonts.ready)
  await page.screenshot({ path: `${OUT}/card_${c.id}.png` })
}
for (const [id, text] of Object.entries(spec.captions)) {
  await page.setContent(captionHtml(text), { waitUntil: 'networkidle' })
  await page.evaluate(() => document.fonts.ready)
  await page.screenshot({ path: `${OUT}/cap_${id}.png`, omitBackground: true })
}
await browser.close()
console.log('cards', spec.cards.length, 'captions', Object.keys(spec.captions).length)
