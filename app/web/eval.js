// How well it works. Draws whatever eval/report.py wrote to eval.json, so a score or a miss it adds shows up here with no change.
// The headline sums the score rows by status. It adds nothing the rows below do not hold.
const heading = key => key.replace(/_/g, ' ')
const num = v => typeof v === 'number' ? v : 0
const has = v => typeof v === 'number'

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

// Sums one score over the rows that match `pick`, for Storey and the baseline, with the denominators the rows carry.
function total(rows, score, pick) {
  const got = rows.filter(r => r.score === score && pick(r))
  return {
    sets: got.length,
    storey: got.reduce((s, r) => s + num(r.Storey), 0), storeyKnown: got.some(r => has(r.Storey)),
    baseline: got.reduce((s, r) => s + num(r.baseline), 0), baselineKnown: got.some(r => has(r.baseline)),
    outOf: got.reduce((s, r) => s + num(r['out of']), 0), baselineOutOf: got.reduce((s, r) => s + num(r['baseline out of']), 0),
    cells: got.reduce((s, r) => s + num(r['absent cells']) + num(r['erased cells']), 0),
  }
}

function card(label, value, detail) {
  return el('div', { className: 'stat' }, el('span', { className: 'label', textContent: label }), el('strong', { textContent: value }), el('span', { className: 'detail', textContent: detail }))
}

function median(values) {
  const v = values.filter(has).sort((a, b) => a - b)
  return v.length ? (v.length % 2 ? v[(v.length - 1) / 2] : (v[v.length / 2 - 1] + v[v.length / 2]) / 2) : null
}

function headline(data) {
  const rows = data.scores || []
  const held = r => String(r.status).includes('held-out'), dev = r => !held(r) && r.status !== 'post hoc'
  const generated = r => /_\d+_(clean|mixed|second|corridor)$/.test(r.set)
  const wrongHeld = total(rows, 'wrong passes', held), wrongDev = total(rows, 'wrong passes', dev)
  const planted = total(rows, 'truth items found', r => generated(r) && dev(r))
  const noCause = total(rows, 'false findings', r => generated(r) && dev(r))
  const cost = median(rows.filter(r => r.score === 'cost USD').map(r => r.Storey)), minutes = median(rows.filter(r => r.score === 'minutes').map(r => r.Storey))
  const cards = [
    card('Wrong passes, held-out sets', wrongHeld.storeyKnown ? String(wrongHeld.storey) : 'Not run', wrongHeld.sets ? `on ${wrongHeld.cells} stair cells truth says are absent or erased` : ''),
    card('Wrong passes, development sets', wrongDev.storeyKnown ? String(wrongDev.storey) : 'Not run', wrongDev.sets ? `on ${wrongDev.cells} absent or erased stair cells and generated stair defects` : ''),
    card('Planted defects raised', planted.outOf ? `${planted.storey} of ${planted.outOf}` : 'Not run', planted.baselineKnown ? `Spark asked directly: ${planted.baseline} of ${planted.baselineOutOf}` : ''),
    card('Findings with no planted cause', noCause.storeyKnown ? String(noCause.storey) : 'Not run', noCause.baselineKnown ? `Spark asked directly: ${noCause.baseline}` : ''),
    card('Cost of one review', cost == null ? 'Not run' : `${cost.toFixed(2)} USD`, minutes == null ? '' : `median, ${minutes.toFixed(1)} minutes, cold cache`),
  ]
  return [el('section', { className: 'stats', 'aria-label': 'Headline numbers' }, ...cards),
    el('p', { className: 'note', textContent: 'Each number is summed from the rows below. Wrong passes count stairs shown Found where truth says they are absent or were erased, and comments shown Looks resolved where truth holds them open.' })]
}

async function start() {
  const results = $('#results')
  try {
    const data = await request('eval.json')
    results.replaceChildren(...headline(data), ...draw(data))
  } catch {
    results.replaceChildren(el('p', { className: 'note', textContent: 'The evaluation has not been published yet.' }))
  }
}
start()
