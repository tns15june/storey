// Helpers every page shares.
const $ = s => document.querySelector(s)
const el = (tag, props = {}, ...kids) => { const e = Object.assign(document.createElement(tag), props); e.append(...kids); return e }
const LOCKED = 'This server needs its access key. Open the link printed in the terminal when Storey started.'
const sheetCount = n => `${n} sheet${n === 1 ? '' : 's'}`

// sample/index.json lists what a static site can show: {"samples": [{"id": "habs_stelizabeths_center", "name": "St Elizabeths, Center Building", "note": "5 floors, scanned", "jurisdictions": ["houston", "philadelphia", "seattle"]}]}. Each id is a folder sample/<id>/ copied from a finished review: review.<jurisdiction>.json, stage_floors.json, sheets.json (only index, width and height are read) and the JPEGs those files name. Any URL under /api/reviews/<id>/files/ is read from that folder, so the JSON needs no rewriting.
const samples = () => request('sample/index.json').then(index => index.samples, () => [])

async function request(url, options) {
  const r = await fetch(url, options)
  if (r.ok) return r.json()
  const detail = await r.json().then(body => body.detail, () => null)
  throw Object.assign(new Error(typeof detail === 'string' ? detail : `The server answered ${r.status}.`), { status: r.status })
}

// Images sit under /api/, so a page awaits this before it loads any. The key leaves the address bar first, and a wrong key shows up as a 401 on the next request.
async function openSession() {
  const key = new URLSearchParams(location.hash.slice(1)).get('key')
  if (!key) return
  history.replaceState(null, '', location.pathname + location.search)
  await fetch('/api/session', { method: 'POST', headers: { 'content-type': 'application/json' }, body: JSON.stringify({ key }) }).catch(() => {})
}

// One viewer's conveniences. A private window can refuse storage, and then nothing is remembered.
const remembered = {
  get(key, fallback) { try { return JSON.parse(localStorage.getItem(key)) ?? fallback } catch { return fallback } },
  set(key, value) { try { localStorage.setItem(key, JSON.stringify(value)) } catch { /* see above */ } },
}

// Sends the chosen files, then opens the new review. Each review spends API credit, so choosing files does not start one.
function wireUpload(form, url) {
  const input = form.querySelector('input'), go = form.querySelector('button'), chosen = form.querySelector('.chosen'), error = form.querySelector('.error')
  const prompt = chosen.textContent, label = go.textContent
  input.onchange = () => {
    const files = input.files
    chosen.textContent = !files.length ? prompt : files.length === 1 ? files[0].name : `${files.length} files`
    go.disabled = !files.length
  }
  form.onsubmit = async e => {
    e.preventDefault()
    const body = new FormData()
    for (const file of input.files) body.append('files', file)
    go.disabled = true; go.textContent = 'Uploading'; error.textContent = ''
    try {
      const { id } = await request(url, { method: 'POST', body })
      location.href = `index.html?review=${id}`
    } catch (err) {
      error.textContent = err.status === 401 ? LOCKED : err.message
      go.disabled = false; go.textContent = label
    }
  }
}
