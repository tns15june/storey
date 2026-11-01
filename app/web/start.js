// Start screen: upload, past reviews and sample reviews. With no server, as on the static demo, only the samples show.
const STATE = { queued: 'Waiting', running: 'Running', done: 'Finished', failed: 'Stopped' }

function fill(list, rows, empty) {
  list.replaceChildren(...(rows.length ? rows : [el('li', { className: 'note', textContent: empty })]))
}

function pastRow(r) {
  const when = new Date(r.updated * 1000).toLocaleString([], { dateStyle: 'medium', timeStyle: 'short' })
  const facts = [r.parent && 'Second review', STATE[r.state], r.sheets && sheetCount(r.sheets), `${(r.usd || 0).toFixed(2)} USD`, when].filter(Boolean).join('  ·  ')
  return el('li', {}, el('a', { href: `index.html?review=${r.id}` }, el('strong', { textContent: r.name || 'Untitled set' }), el('span', { textContent: facts })))
}

function sampleRow(s) {
  return el('li', {}, el('a', { href: `index.html?sample=${encodeURIComponent(s.id)}` }, el('strong', { textContent: s.name }), el('span', { textContent: s.note || '' })))
}

async function start() {
  await openSession()
  fill($('#samples'), (await samples()).map(sampleRow), 'No sample reviews are published yet.')
  const r = await fetch('/api/reviews').catch(() => null)
  if (r?.status === 401) {
    $('#locked').textContent = LOCKED
    $('#locked').hidden = false
  } else if (r?.ok) {
    document.body.dataset.mode = 'live'
    wireUpload($('#upload'), '/api/reviews')
    fill($('#past'), (await r.json()).map(pastRow), 'No reviews yet. Upload a set to start one.')
  }
}
start()
