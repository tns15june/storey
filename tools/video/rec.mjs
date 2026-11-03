// Records one shot of the running app at 1920x1080 with a visible cursor, and the time of each step in <shot>.json.
//   node rec.mjs <shot> <out dir> <reviews.json>
// Shots: upload, stairs, test, findings, city, second, letter, eval. reviews.json names the server's review ids:
//   {"stel": St. Elizabeths, "capitol": The Capitol, "mixed": the generated 710 set, "second": its cycle 2}
// The server runs from the repo with STOREY_ACCESS_KEY set; STOREY_URL defaults to http://127.0.0.1:8811.
import fs from 'fs'
import { chromium } from 'playwright'   // npm install playwright, in this folder

const [shot, OUT, REVIEWS] = process.argv.slice(2)
const BASE = process.env.STOREY_URL || 'http://127.0.0.1:8811', KEY = process.env.STOREY_ACCESS_KEY
const IDS = JSON.parse(fs.readFileSync(REVIEWS, 'utf8'))
const CAPITOL_PDF = new URL('../../data/plans/mml_capitol/THE-CAPITOL.pdf', import.meta.url).pathname   // the public pattern-book set, not in git

const CURSOR = `
addEventListener('DOMContentLoaded', () => {
  const c = document.createElement('div')
  c.id = 'fakecursor'
  Object.assign(c.style, { position: 'fixed', left: '-40px', top: '-40px', width: '22px', height: '22px', marginLeft: '-11px', marginTop: '-11px', borderRadius: '50%',
    background: 'rgba(196,37,47,.35)', border: '2px solid #c4252f', zIndex: 2147483647, pointerEvents: 'none', transition: 'transform .12s' })
  document.body.append(c)
  addEventListener('mousemove', e => { c.style.left = e.clientX + 'px'; c.style.top = e.clientY + 'px' }, true)
  addEventListener('mousedown', () => { c.style.transform = 'scale(.6)' }, true)
  addEventListener('mouseup', () => { c.style.transform = 'scale(1)' }, true)
})`

const browser = await chromium.launch({ channel: 'chrome' })
const ctx = await browser.newContext({ viewport: { width: 1920, height: 1080 }, recordVideo: { dir: OUT, size: { width: 1920, height: 1080 } } })
await ctx.addInitScript(CURSOR)
const page = await ctx.newPage()
const errors = []
page.on('pageerror', e => errors.push(String(e)))
const wait = ms => page.waitForTimeout(ms)
let at = { x: 960, y: 540 }
async function glide(x, y, steps = 30) { await page.mouse.move(x, y, { steps }); at = { x, y } }
async function click(selector, { pause = 500, nth = 0 } = {}) {
  const el = page.locator(selector).nth(nth)
  await el.scrollIntoViewIfNeeded()
  const b = await el.boundingBox()
  await glide(b.x + b.width / 2, b.y + b.height / 2)
  await wait(pause)
  await page.mouse.down(); await wait(90); await page.mouse.up()
}
async function open(path) {
  await page.goto(BASE + path)
  await page.mouse.move(at.x, at.y)
}
const marks = []
const mark = name => marks.push({ name, t: Date.now() })

// The session cookie first, so every page can read the reviews.
await page.goto(BASE + '/start.html#key=' + KEY)
await wait(1200)
const t0 = Date.now()
mark('start')

if (shot === 'stairs') {
  await open('/index.html?review=' + IDS.stel); await wait(3500); mark('loaded')
  await click('#play'); mark('walk'); await wait(9000)
  await click('#play'); await wait(1200)
  if (await page.locator('#matrix td.check button').count()) { await click('#matrix td.check button'); mark('cell'); await wait(5000) }
  await click('#matrix td.found button'); mark('found'); await wait(5000)
  mark('end')
} else if (shot === 'findings') {
  await open('/index.html?review=' + IDS.capitol); await wait(3500); mark('loaded')
  await click('#panel-findings article.finding button.head', { nth: 0 }); mark('first'); await wait(4500)
  await click('#panel-findings article.finding button.head', { nth: 1 }); mark('second'); await wait(4500)
  await click('#tab-not'); mark('notchecked'); await wait(4000)
  mark('end')
} else if (shot === 'city') {
  await open('/index.html?review=' + IDS.mixed); await wait(3500); mark('loaded')
  const cards = page.locator('#panel-findings article.finding')
  await cards.nth(12).scrollIntoViewIfNeeded(); await wait(400)
  let b = await cards.nth(11).boundingBox(); await glide(b.x + b.width / 2, b.y + 30); mark('houston'); await wait(3500)
  await click('#jurisdiction'); await wait(300)
  await page.selectOption('#jurisdiction', 'seattle'); mark('seattle'); await wait(1800)
  await cards.nth(11).scrollIntoViewIfNeeded(); await wait(400)
  b = await cards.nth(10).boundingBox(); await glide(b.x + b.width / 2, b.y + 30); await wait(4000)
  await page.selectOption('#jurisdiction', 'houston'); await wait(1500)   // leave the review as it was
  mark('end')
} else if (shot === 'second') {
  await open('/index.html?review=' + IDS.second); await wait(3500); mark('loaded')
  await click('#open-second'); mark('dialog'); await wait(3500)
  await glide(960, 640)
  for (let i = 0; i < 7; i++) { await page.mouse.wheel(0, 300); await wait(1000) }
  mark('end')
} else if (shot === 'letter') {
  await open('/index.html?review=' + IDS.mixed); await wait(3500); mark('loaded')
  await click('#open-letter'); mark('letter'); await wait(3000)
  for (let i = 0; i < 4; i++) { await page.mouse.wheel(0, 300); await wait(900) }
  mark('end')
} else if (shot === 'upload') {
  await open('/start.html'); await wait(2500); mark('loaded')
  const label = page.locator('#upload .drop'); const lb = await label.boundingBox()
  await glide(lb.x + lb.width / 2, lb.y + lb.height / 2); await wait(600)
  await page.setInputFiles('#upload input[type=file]', CAPITOL_PDF); mark('chosen'); await wait(1500)
  await click('#upload button'); mark('started')
  await page.waitForFunction(() => document.body.dataset.view === 'review', null, { timeout: 480000, polling: 1000 }); mark('done')
  await wait(4000); mark('end')
} else if (shot === 'test') {
  await open('/index.html?review=' + IDS.stel); await wait(3500); mark('loaded')
  await click('#matrix button[data-vertical="C"][data-sheet="3"]'); mark('cell'); await wait(3000)
  await click('#cell .ghost.test'); mark('removed')
  await page.waitForURL(u => !u.href.includes(IDS.stel), { timeout: 60000 }); await wait(1000)
  await page.waitForFunction(() => document.body.dataset.view === 'review', null, { timeout: 480000, polling: 1000 }); mark('done')
  await wait(2500)
  const card = page.locator('#panel-findings article.finding', { hasText: 'Stair C' }).first()
  if (await card.count()) { await card.scrollIntoViewIfNeeded(); await click('#panel-findings article.finding:has-text("Stair C") button.head'); mark('finding'); await wait(6000) }
  mark('end')
} else if (shot === 'eval') {
  await open('/eval.html'); await wait(6000); mark('loaded')
  mark('end')
}

const video = page.video()
await ctx.close()
const path = await video.path()
fs.renameSync(path, `${OUT}/${shot}.webm`)
fs.writeFileSync(`${OUT}/${shot}.json`, JSON.stringify({ marks: marks.map(m => ({ name: m.name, s: (m.t - t0) / 1000 })), errors }, null, 1))
console.log(shot, JSON.stringify(errors))
await browser.close()
