// How well it works. Draws whatever eval/report.py wrote to eval.json, so a score or a miss it adds shows up here with no change.
const heading = key => key.replace(/_/g, ' ')

function table(rows) {
  const keys = [...new Set(rows.flatMap(Object.keys))]
  const text = value => value == null ? '' : typeof value === 'object' ? JSON.stringify(value) : String(value)
  return el('div', { className: 'scroll-x' }, el('table', {},
    el('thead', {}, el('tr', {}, ...keys.map(k => el('th', { scope: 'col', textContent: heading(k) })))),
    el('tbody', {}, ...rows.map(row => el('tr', {}, ...keys.map(k => el('td', { textContent: text(row[k]) })))))))
}

function draw(value, depth = 2) {
  if (Array.isArray(value)) {
    if (!value.length) return [el('p', { className: 'note', textContent: 'None.' })]
    return value.every(row => row && typeof row === 'object' && !Array.isArray(row)) ? [table(value)] : [el('ul', { className: 'lines' }, ...value.map(line => el('li', { textContent: String(line) })))]
  }
  if (value && typeof value === 'object') return Object.entries(value).flatMap(([k, v]) => [el(`h${depth}`, { textContent: heading(k) }), ...draw(v, Math.min(depth + 1, 3))])
  return [el('p', { textContent: value == null ? 'Not available.' : String(value) })]
}

async function start() {
  const results = $('#results')
  try {
    results.replaceChildren(...draw(await request('eval.json')))
  } catch {
    results.replaceChildren(el('p', { className: 'note', textContent: 'The evaluation has not been published yet.' }))
  }
}
start()
