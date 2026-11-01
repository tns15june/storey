// Review screen. Follows a review through its stages, then draws its JSON. All state lives here, so a new review only needs new JSON.
const SEVERITY = { fix: 'Must fix', check: 'Check', unknown: 'Cannot tell' }
const CELL = { found: 'Found', missing: 'Missing', check: 'Check', not_followed: 'Not followed', end: 'Not on this floor' }
const RECHECK = { resolved_looks: 'Looks resolved', open: 'Still open', unknown: 'Cannot tell', new_change: 'New change' }
const STAGE = { queued: 'Waiting', running: 'Running', done: 'Done', failed: 'Did not finish' }
const ALL_DROPPED = 'The examiner dropped every comment, so this letter lists none.'
const EDITS_HIDDEN = 'Edits made under other selections are hidden. Change the selection back to see them.'
const STAIR_PENS = 8      // --stair-1 to --stair-8 in style.css
const POLL = 2000
const FADE = 320          // ms, the .layer transition in style.css
const WALK = 1200         // ms on each floor
const query = new URLSearchParams(location.search)
const REVIEW = query.get('review')?.match(/^[0-9a-f]{32}$/)?.[0]
const SAMPLE = !REVIEW && query.get('sample')?.match(/^[\w-]+$/)?.[0]
const BASE = REVIEW ? `/api/reviews/${REVIEW}/files/` : `sample/${SAMPLE}/`
const NS = 'http://www.w3.org/2000/svg'
const svgEl = (tag, attrs, text) => { const e = document.createElementNS(NS, tag); for (const k in attrs) e.setAttribute(k, attrs[k]); if (text != null) e.textContent = text; return e }

const state = { review: null, status: null, sheet: null, active: null, spot: null, cell: null, showStairs: true, showFindings: true, scale: 1, x: 0, y: 0, walking: null, climb: 1, placement: {}, reference: null, labelled: null }
const layers = {}
let writes = Promise.resolve()   // one write at a time, because each answer replaces the whole review
let layerOrder = 1

const sheetOf = id => state.review.sheets.find(s => s.id === id)
const nameOf = s => s ? (s.number && s.number !== s.level ? `${s.number} ${s.level}` : s.level) : 'Whole set'
const pen = i => `var(--stair-${i % STAIR_PENS + 1})`
const titleCase = text => text ? text[0].toUpperCase() + text.slice(1).replace(/_/g, ' ') : ''
const pill = (text, kind) => el('span', { className: `pill ${kind}`, textContent: text })

function say(message) {
  $('#notice').textContent = message
  $('#notice').hidden = !message
}

function post(path, body) {
  const send = () => request(`/api/reviews/${REVIEW}/${path}`, { method: 'POST', headers: { 'content-type': 'application/json' }, body: JSON.stringify(body) })
  const answer = writes.then(send)
  writes = answer.catch(() => {})
  return answer
}

// A revision cloud: the scalloped outline examiners draw around a problem on a sheet.
function cloudPath([x1, y1, x2, y2], r) {
  const side = (ax, ay, bx, by) => {
    const len = Math.hypot(bx - ax, by - ay), n = Math.max(2, Math.round(len / (r * 1.7)))
    let d = ''
    for (let i = 1; i <= n; i++) d += ` A ${r} ${r} 0 0 1 ${ax + (bx - ax) * i / n} ${ay + (by - ay) * i / n}`
    return d
  }
  return `M ${x1} ${y1}` + side(x1, y1, x2, y1) + side(x2, y1, x2, y2) + side(x2, y2, x1, y2) + side(x1, y2, x1, y1) + ' Z'
}

// Registration is not in review.json. The floors stage keeps it in full sheet pixels and the screen draws on 2000 px JPEGs, so each matrix is converted here.
async function loadRegistration(review) {
  const [floors, rows] = await Promise.all([request(BASE + 'stage_floors.json'), request(BASE + 'sheets.json')]).catch(() => [{}, []])
  const size = Object.fromEntries(rows.map(row => [row.index, row]))
  const shown = Object.fromEntries(review.sheets.map(s => [s.id, s]))
  state.labelled = floors.discharge ?? null
  state.reference = shown[floors.reference] && floors.transforms?.[floors.reference] ? String(floors.reference) : null   // a set of one floor has a reference and no transform, and nothing to line up with
  state.placement = {}
  if (state.reference == null) return
  const rx = shown[state.reference].w / size[state.reference].width, ry = shown[state.reference].h / size[state.reference].height
  for (const [id, [[a, c, e], [b, d, f]]] of Object.entries(floors.transforms || {})) {
    const sx = shown[id].w / size[id].width, sy = shown[id].h / size[id].height
    state.placement[id] = [rx * a / sx, ry * b / sx, rx * c / sy, ry * d / sy, rx * e, ry * f]
  }
}

function toStage(id, x, y) {
  const m = state.placement[id]
  return m ? [m[0] * x + m[2] * y + m[4], m[1] * x + m[3] * y + m[5]] : [x, y]
}

// Floors share the reference sheet's frame, so a change of floor keeps the pan and zoom and the building stays put.
const frameOf = id => state.reference != null && state.review.floor_order.includes(id) ? state.reference : id

function renderMeta() {
  const p = state.review.project
  document.title = `${p.name || 'Review'} · Storey`
  $('#project').textContent = $('#project').title = p.name || 'Untitled set'
  $('#address').textContent = p.address || ''
  $('#address-cell').hidden = !p.address
  $('#cycle').textContent = `Cycle ${p.cycle}`
  $('#code').textContent = $('#code').title = p.code
  $('#jurisdiction').value = p.jurisdiction
  $('#sprinklers').value = p.sprinklers || ''
  const floors = [...state.review.floor_order].reverse().map(id => el('option', { value: id, textContent: sheetOf(id).level }))
  $('#discharge').replaceChildren(...(state.labelled == null ? [el('option', { value: '', textContent: 'Not settled' })] : []), ...floors)
  $('#discharge').value = p.discharge_level ?? ''
  $('#discharge').disabled = !floors.length
  const failed = (state.status?.stages || []).filter(row => row.state === 'failed').map(row => row.label)
  $('#failed').hidden = !failed.length
  $('#failed').replaceChildren(`These stages did not finish: ${failed.join(', ')}. `, el('button', { className: 'link', textContent: 'See Not checked', onclick: () => selectTab('not') }))
  $('#note').textContent = p.note || ''
  $('#note').hidden = !p.note
}

function renderFloors() {
  const { floor_order, sheets, findings } = state.review
  const nav = $('#floors')
  $('#floors-title').textContent = floor_order.length ? 'Floors' : 'Sheets'
  nav.setAttribute('aria-label', floor_order.length ? 'Floors, top to bottom' : 'Sheets')
  nav.querySelectorAll('.floor').forEach(n => n.remove())
  for (const s of floor_order.length ? [...floor_order].reverse().map(sheetOf) : sheets) {
    const n = findings.filter(f => f.sheet === s.id && f.severity !== 'unknown').length
    const b = el('button', { className: 'floor', onclick: () => go(s.id) },
      el('img', { src: s.image, alt: '', loading: 'lazy' }), el('span', { className: 'name', textContent: s.level, title: s.level }))
    b.dataset.id = s.id
    if (n) b.append(el('span', { className: 'count', textContent: n, title: `${n} on this floor` }))
    nav.append(b)
  }
}

function signalLine(cell) {
  const { spark, crop, ink, aligned, reason } = cell.signals || {}
  return [spark && `Runs ${spark}`, crop && `crop answer ${crop}`, ink != null && `ink ${ink.toFixed(2)}`, aligned === false && 'floor not aligned', reason].filter(Boolean).join(', ')
}

function matrixCell(v, s, split) {
  const cell = v.floors[s.id]
  if (!cell) return el('td', { className: split ? 'split' : '' })
  const comments = state.review.findings.filter(f => f.vertical === v.id && f.sheet === s.id).map(f => f.id)
  const label = [`Stair ${v.id}, ${s.level}: ${CELL[cell.state]}`, signalLine(cell), comments.length && `Comment ${comments.join(', ')}`].filter(Boolean).join('. ')
  const b = el('button', { title: label, onclick: () => openCell(v, s) })
  b.setAttribute('aria-label', label)
  b.dataset.vertical = v.id
  b.dataset.sheet = s.id
  return el('td', { className: [cell.state, comments.length && 'flagged', split && 'split'].filter(Boolean).join(' ') }, b)
}

function renderMatrix() {
  const { verticals, floor_order } = state.review
  const followed = verticals.filter(v => v.established), seen = verticals.filter(v => !v.established), columns = [...followed, ...seen]
  const floors = [...floor_order].reverse().map(sheetOf)
  const empty = !floors.length ? 'No floors were put in order, so no stairs were followed. See Not checked.' : !columns.length ? 'Stairs were not followed on these floors. See Not checked.' : ''
  $('#matrix-empty').textContent = empty
  $('#matrix-empty').hidden = !empty
  $('#legend').hidden = !!empty
  if (empty) return $('#matrix').replaceChildren()
  const group = (name, rows, className) => rows.length ? [el('th', { scope: 'colgroup', colSpan: rows.length, textContent: name, className })] : []
  const headings = el('tr', {}, el('td'), ...group('Followed', followed, 'group'), ...group('Seen by one run', seen, `group${followed.length ? ' split' : ''}`))
  const head = el('tr', {}, el('td'), ...columns.map((v, i) => {
    const th = el('th', { scope: 'col', textContent: v.id, title: v.name || `Stair ${v.id}`, className: i === followed.length && i ? 'split' : '' })
    th.style.setProperty('--pen', pen(verticals.indexOf(v)))
    return th
  }))
  const rows = floors.map(s => {
    const tr = el('tr', {}, el('th', { scope: 'row', className: 'row', textContent: s.level, title: s.level }), ...columns.map((v, i) => matrixCell(v, s, i === followed.length && i)))
    tr.dataset.id = s.id
    return tr
  })
  $('#matrix').replaceChildren(el('thead', {}, headings, head), el('tbody', {}, ...rows))
  markCell()
}

function markCell() {
  document.querySelectorAll('.matrix td button').forEach(b => b.setAttribute('aria-pressed', b.dataset.vertical === state.cell?.vertical && b.dataset.sheet === state.cell?.sheet))
}

// The server's own test in storey/testtool.py: the places a finding can appear.
function removable(v, s) {
  const order = state.review.floor_order, here = order.indexOf(s.id)
  const found = order.flatMap((id, at) => v.floors[id]?.state === 'found' ? [at] : [])
  return !SAMPLE && found.includes(here) && ((found[0] < here && here < found.at(-1)) || (s.id === state.review.project.discharge_level && found.at(-1) > here))
}

async function removeStair(v, s, button) {
  button.disabled = true
  say('')
  try {
    const { id } = await post('remove', { vertical: v.id, sheet: Number(s.id) })
    location.href = `index.html?review=${id}`
  } catch (e) {
    say(e.message)
    button.disabled = false
  }
}

function renderCell() {
  const box = $('#cell'), v = state.review.verticals.find(row => row.id === state.cell?.vertical), s = state.cell && sheetOf(state.cell.sheet), cell = v?.floors[state.cell.sheet]
  box.hidden = !cell
  if (!cell) return
  const signals = cell.signals || {}
  const facts = [['Reason', signals.reason], ['Runs', signals.spark], ['Crop answer', signals.crop], ['Ink', signals.ink?.toFixed(2)], ['Aligned', signals.aligned === false ? 'No' : null]]
  const dl = el('dl', {})
  for (const [k, value] of facts) if (value) dl.append(el('dt', { textContent: k }), el('dd', { textContent: value }))
  box.replaceChildren(
    el('div', { className: 'cellhead' }, el('h3', { textContent: `Stair ${v.id}, ${s.level}` }), pill(CELL[cell.state], cell.state),
      el('button', { className: 'ghost small', textContent: 'Close', onclick: () => closeCell() })),
    ...(v.name ? [el('p', { className: 'note', textContent: v.name })] : []),
    cell.crop ? el('img', { className: 'crop', src: cell.crop, alt: `Stair ${v.id}, ${s.level}, crop` }) : el('p', { className: 'note', textContent: 'No crop was taken at this place.' }),
    dl)
  if (removable(v, s)) box.append(el('button', { className: 'ghost test', textContent: 'Test: remove this stair from the sheet', onclick: e => removeStair(v, s, e.currentTarget) }))
}

function closeCell() {
  state.cell = null
  markCell()
  renderCell()
}

function openCell(v, s) {
  state.cell = { vertical: v.id, sheet: s.id }
  markCell()
  renderCell()
  const point = v.floors[s.id].point
  if (point) lookAt(s.id, [point[0] - 65, point[1] - 65, point[0] + 65, point[1] + 65])
  else go(s.id)
}

function setDecision(f, choice, paint) {
  const decisions = state.review.decisions, before = decisions[f.key] ?? null, next = before === choice ? null : choice
  decisions[f.key] = next
  paint()
  if (SAMPLE) return remembered.set(`storey.decisions.${SAMPLE}`, decisions)
  say('')
  post('decisions', { key: f.key, decision: next }).then(review => { state.review = review }, e => {
    state.review.decisions[f.key] = before
    paint()
    say(e.message)
  })
}

function findingCard(f) {
  const where = f.sheet == null ? null : sheetOf(f.sheet)
  const delta = svgEl('svg', { class: 'delta', viewBox: '0 0 44 36', role: 'img', 'aria-label': `Comment ${f.id}` })
  delta.append(svgEl('path', { d: 'M22 2 L42 34 H2 Z' }), svgEl('text', { x: 22, y: 30 }, f.id))
  const head = el('button', { className: 'head', onclick: () => focusItem(f) }, delta,
    el('div', {}, pill(SEVERITY[f.severity], f.severity), ...(f.state ? [pill(RECHECK[f.state], `recheck ${f.state}`)] : []), el('h3', { textContent: f.title }),
      el('div', { className: 'where', textContent: [nameOf(where), f.rule].filter(Boolean).join('  ·  ') })))
  const dl = el('dl', {})
  for (const [k, v] of [['Found', f.measured], ['Needed', f.required], ['Why we think so', f.evidence]]) if (v) dl.append(el('dt', { textContent: k }), el('dd', { textContent: v }))
  const choices = [['keep', 'Keep'], ['drop', 'Drop']].map(([value, textContent]) => el('button', { value, textContent }))
  const card = el('article', { className: `finding ${f.severity}` }, head, ...(f.crop ? [el('img', { className: 'crop', src: f.crop, alt: `Drawing at comment ${f.id}` })] : []), dl, el('div', { className: 'decide' }, ...choices))
  const paint = () => {
    const decision = state.review.decisions[f.key]
    card.classList.toggle('dropped', decision === 'drop')
    choices.forEach(b => b.setAttribute('aria-pressed', decision === b.value))
  }
  choices.forEach(b => { b.onclick = () => setDecision(f, b.value, paint) })
  paint()
  card.dataset.key = f.key
  card.setAttribute('aria-current', state.active === f.key)
  return card
}

function renderFindings() {
  const { findings } = state.review
  $('#panel-findings').replaceChildren(...(findings.length ? findings.map(findingCard) : [el('p', { className: 'note', textContent: 'No findings were raised. That does not mean this set has no problems. See Not checked.' })]))
}

function renderSet() {
  const { sheets, cost } = state.review
  const reviewed = sheets.filter(s => s.reviewed).length
  const rows = sheets.map(s => {
    const facts = [titleCase(s.class), s.kind, s.scale].filter(Boolean).join('  ·  ')
    return el('li', {}, el('button', { className: `row${s.reviewed ? '' : ' unreviewed'}`, onclick: () => focusItem({ sheet: s.id }) },
      el('span', { className: 'number', textContent: s.number || 'No number' }), el('span', { className: 'title', textContent: s.title || 'No title read' }),
      el('span', { className: 'facts', textContent: s.reviewed ? facts : s.reason || 'Not reviewed' })))
  })
  const summary = sheets.length ? `${sheetCount(sheets.length)}, ${reviewed} reviewed. This review took ${Math.round(cost.seconds)} seconds and ${cost.usd.toFixed(2)} USD.` : 'No sheets were read.'
  $('#panel-set').replaceChildren(el('p', { className: 'note', textContent: summary }), el('ul', { className: 'rows' }, ...rows))
}

function renderOpen() {
  const items = state.review.open_items
  const rows = items.map(item => {
    const li = el('li', {}, el('button', { className: 'row', onclick: () => focusItem(item) },
      el('span', { className: 'title', textContent: item.title }), el('span', { className: 'facts', textContent: [item.sheet != null && nameOf(sheetOf(item.sheet)), item.evidence].filter(Boolean).join('  ·  ') })))
    li.firstChild.dataset.key = item.key
    return li
  })
  $('#panel-open').replaceChildren(...(items.length ? [el('ul', { className: 'rows' }, ...rows)] : [el('p', { className: 'note', textContent: 'No open items were raised. See Not checked.' })]))
}

function renderTabs() {
  const { findings, sheets, open_items, not_checked } = state.review
  for (const [name, list] of [['findings', findings], ['set', sheets], ['open', open_items], ['not', not_checked]]) $(`#tab-${name} .n`).textContent = list.length
  $('#not-checked').replaceChildren(...(not_checked.length ? not_checked.map(line => el('li', { textContent: line })) : [el('li', { className: 'note', textContent: 'No checks were skipped.' })]))
}

function selectTab(name, focus) {
  document.querySelectorAll('[role="tab"]').forEach(tab => {
    const on = tab.id === `tab-${name}`
    tab.setAttribute('aria-selected', on)
    tab.tabIndex = on ? 0 : -1
    $(`#${tab.getAttribute('aria-controls')}`).hidden = !on
    if (on && focus) tab.focus()
  })
}

function stairMarks(s, u) {
  const marks = []
  const label = (x, y, text, color) => marks.push(svgEl('text', { x, y, class: 'tag', style: `fill:${color};font-size:${15 * u}px;stroke-width:${3.5 * u}px` }, text))
  state.review.verticals.forEach((v, i) => {
    const cell = v.floors[s.id], color = pen(i)
    if (!cell) return
    if (cell.state === 'found' && cell.mask) {
      marks.push(svgEl('polygon', { points: cell.mask.join(' '), style: `fill:${color};fill-opacity:.32;stroke:${color};stroke-width:${1.5 * u}` }))
      const xs = cell.mask.map(p => p[0]), ys = cell.mask.map(p => p[1])
      label(Math.min(...xs), Math.min(...ys) - 5 * u, v.id, color)
    } else if (cell.point && ['found', 'check', 'missing'].includes(cell.state)) {
      const ring = { found: color, check: 'var(--pen-check)', missing: 'var(--pen-fix)' }[cell.state], [x, y] = cell.point
      marks.push(svgEl('circle', { cx: x, cy: y, r: 12 * u, style: `fill:none;stroke:${ring};stroke-width:${2 * u};stroke-dasharray:${cell.state === 'check' ? `${5 * u} ${3 * u}` : 'none'}` }))
      label(x + 15 * u, y - 8 * u, v.id, ring)
    }
  })
  return marks
}

function routeMarks(s, u) {
  return state.review.routes.filter(r => r.sheet === s.id && r.path.length).flatMap(r => {
    const [x, y] = r.path[Math.floor(r.path.length / 2)], right = x > s.w / 2   // the label runs toward the middle of the sheet, or it leaves a phone's screen
    return [svgEl('polyline', { points: r.path.join(' '), class: 'route', style: `stroke-width:${2.5 * u};stroke-dasharray:${8 * u} ${5 * u}` }),
      svgEl('text', { x: x + (right ? -8 : 8) * u, y: y - 8 * u, 'text-anchor': right ? 'end' : 'start', class: 'tag route-label', style: `font-size:${13 * u}px;stroke-width:${3.5 * u}px` }, `${r.length} ft. ${r.label}`)]
  })
}

function cloud([x1, y1, x2, y2], color, u, active) {
  const pad = 8 * u
  return svgEl('path', { d: cloudPath([x1 - pad, y1 - pad, x2 + pad, y2 + pad], 7 * u), style: `fill:${active ? color : 'none'};fill-opacity:.12;stroke:${color};stroke-width:${2 * u}` })
}

function findingMarks(s, u) {
  const { spot } = state, marked = spot?.box && spot.sheet === s.id && !spot.severity   // a finding has its own cloud
  return [...(marked ? [cloud(spot.box, 'var(--pen-check)', u, true)] : []), ...state.review.findings.filter(f => f.sheet === s.id && f.box).flatMap(f => {
    const color = `var(--pen-${f.severity})`, tx = f.box[2] + 22 * u, ty = f.box[1] - 6 * u, w = 21 * u
    return [cloud(f.box, color, u, state.active === f.key),
      svgEl('path', { d: `M ${tx} ${ty - 18 * u} L ${tx + w} ${ty + 16 * u} H ${tx - w} Z`, style: `fill:var(--paper);stroke:${color};stroke-width:${1.5 * u}` }),
      svgEl('text', { x: tx, y: ty + 12 * u, 'text-anchor': 'middle', style: `fill:${color};font:600 ${10.5 * u}px var(--narrow)` }, f.id)]
  })]
}

// One layer per sheet: its image and overlay in that sheet's own pixels, placed by its registration, so the floors and their stair masks line up.
function layerOf(s) {
  if (layers[s.id]) return layers[s.id]
  const img = el('img', { src: s.image, alt: `Sheet ${[s.number, s.title].filter(Boolean).join(', ') || s.id}`, draggable: false, width: s.w, height: s.h })
  const svg = svgEl('svg', { viewBox: `0 0 ${s.w} ${s.h}`, 'aria-hidden': 'true' })
  const box = el('div', { className: 'layer' }, img, svg)
  box.style.width = `${s.w}px`
  box.style.height = `${s.h}px`
  if (state.placement[s.id]) box.style.transform = `matrix(${state.placement[s.id]})`
  $('#stage').append(box)
  return layers[s.id] = { box, img, svg }
}

// The new floor fades in over the old one, so the page never shows through between floors.
function showLayer(layer) {
  if (layer.box.classList.contains('on') && Number(layer.box.style.zIndex) === layerOrder) return
  layer.box.style.zIndex = ++layerOrder
  layer.box.classList.add('on')
  clearTimeout(showLayer.timer)
  showLayer.timer = setTimeout(() => Object.values(layers).forEach(l => { if (l !== layer) l.box.classList.remove('on') }), FADE)
}

function renderSheet() {
  const s = sheetOf(state.sheet)
  if (!s) return $('#floorname').textContent = 'No sheets to show'
  const layer = layerOf(s), u = 1 / state.scale
  layer.svg.replaceChildren(...(state.showStairs ? stairMarks(s, u) : []), ...(state.showFindings ? [...routeMarks(s, u), ...findingMarks(s, u)] : []))
  showLayer(layer)
  const aligned = !state.review.floor_order.includes(s.id) || state.reference == null || state.placement[s.id]
  $('#floorname').textContent = [nameOf(s), s.title, !aligned && 'Not lined up with the other floors', !s.reviewed && 'Not reviewed'].filter(Boolean).join('  ·  ')
  $('#floorname').title = $('#floorname').textContent
  document.querySelectorAll('#matrix tbody tr').forEach(r => r.classList.toggle('here', r.dataset.id === s.id))
  const nav = $('#floors')
  document.querySelectorAll('.floor').forEach(b => {
    const current = b.dataset.id === s.id
    if (current && b.getAttribute('aria-current') !== 'true') nav.scrollTo({ top: b.offsetTop - (nav.clientHeight - b.offsetHeight) / 2, left: b.offsetLeft - (nav.clientWidth - b.offsetWidth) / 2 })
    b.setAttribute('aria-current', current)
  })
}

function applyView(ease) {
  const stage = $('#stage')
  stage.classList.toggle('ease', !!ease)
  stage.style.transform = `translate(${state.x}px, ${state.y}px) scale(${state.scale})`
}
function fit() {
  const s = sheetOf(frameOf(state.sheet)), v = $('#viewer').getBoundingClientRect()
  if (s && v.width) {
    state.scale = Math.min(v.width / s.w, v.height / s.h) * .97
    state.x = (v.width - s.w * state.scale) / 2; state.y = (v.height - s.h * state.scale) / 2
    applyView()
  }
  renderSheet()
}
function zoomAt(cx, cy, k) {
  const next = Math.min(4, Math.max(.1, state.scale * k)), r = next / state.scale
  state.x = cx - (cx - state.x) * r; state.y = cy - (cy - state.y) * r; state.scale = next
  applyView(); renderSheet()
}
function go(id) {
  const frame = state.sheet != null && frameOf(state.sheet)
  state.sheet = id
  if (frame === frameOf(id)) renderSheet()
  else fit()
}
function lookAt(id, [x1, y1, x2, y2]) {
  go(id)
  const v = $('#viewer').getBoundingClientRect(), [ax, ay] = toStage(id, x1, y1), [bx, by] = toStage(id, x2, y2)
  state.scale = Math.min(1.2, Math.min(v.width, v.height) / (Math.max(Math.abs(bx - ax), Math.abs(by - ay)) * 3.2))
  state.x = v.width / 2 - (ax + bx) / 2 * state.scale; state.y = v.height / 2 - (ay + by) / 2 * state.scale
  applyView(true); renderSheet()
}
function focusItem(item) {
  state.active = item.key ?? null   // a second-review change has no key, and neither has a row of the set
  state.spot = item
  document.querySelectorAll('.finding, .rows .row').forEach(c => c.setAttribute('aria-current', c.dataset.key === state.active))
  if (item.sheet == null) return renderSheet()
  if (matchMedia('(max-width: 960px)').matches) $('#viewer').scrollIntoView({ block: 'center' })
  if (item.box) lookAt(item.sheet, item.box)
  else go(item.sheet)
}
function stepFloor(d) {
  const { floor_order, sheets } = state.review
  const order = floor_order.length ? floor_order : sheets.map(s => s.id).reverse()
  const next = order[Math.min(order.length - 1, Math.max(0, order.indexOf(state.sheet) + d))]
  if (next != null) go(next)
}

function walkOn() {
  const order = state.review.floor_order, at = order.indexOf(state.sheet)
  if (order[at + state.climb] == null) state.climb = -state.climb
  go(order[at + state.climb])
}
async function toggleWalk() {
  const b = $('#play'), order = state.review.floor_order
  if (state.walking) {
    clearInterval(state.walking); state.walking = null
    b.setAttribute('aria-pressed', false); b.textContent = 'Walk the floors'
    return
  }
  b.disabled = true
  await Promise.allSettled(order.map(id => layerOf(sheetOf(id)).img.decode()))   // a floor still decoding would show as a blank step
  b.disabled = false
  b.setAttribute('aria-pressed', true); b.textContent = 'Stop'
  state.climb = 1
  go(order[0])
  state.walking = setInterval(walkOn, WALK)
}

function renderLetter() {
  const { letter, project, decisions } = state.review
  const kept = letter.items.filter(item => decisions[item.key] !== 'drop')   // a sample has no server to filter them
  const note = letter.failed || letter.note || (letter.items.length && !kept.length ? ALL_DROPPED : null)
  // shortcut: edits live in this browser, one per comment, so no other examiner or device sees them and a new edit replaces one hidden under another selection, save them on the server if examiners share a review.
  const key = `storey.edits.${REVIEW || SAMPLE}`, edits = remembered.get(key, {})
  const hidden = Object.entries(edits).some(([k, saved]) => saved.text !== saved.from && saved.from !== kept.find(row => row.key === k)?.text && decisions[k] !== 'drop')
  const fields = el('dl', { className: 'fields' })
  for (const [k, v] of [['Project', project.name || 'Untitled set'], ['Review', `Cycle ${project.cycle}`], ['Code', project.code]]) fields.append(el('div', {}, el('dt', { textContent: k }), el('dd', { textContent: v })))
  const item = row => {
    const saved = edits[row.key]
    // An edit stands only over the text it was made from, because a change of jurisdiction rewrites the section the comment cites.
    const text = el('div', { className: 'text', textContent: saved?.from === row.text ? saved.text : row.text })
    text.setAttribute('contenteditable', 'plaintext-only')
    text.setAttribute('role', 'textbox')
    text.setAttribute('aria-multiline', 'true')
    text.setAttribute('aria-label', `Comment ${row.comment_id}`)
    text.oninput = () => { edits[row.key] = { from: row.text, text: text.textContent }; remembered.set(key, edits) }
    return el('li', {}, el('div', { className: 'itemhead' }, el('span', { className: 'id', textContent: row.comment_id }), ...(row.severity ? [pill(SEVERITY[row.severity], row.severity)] : [])), text)
  }
  const sheets = [...Map.groupBy(kept, row => row.sheet)].map(([heading, rows]) => el('section', {}, el('h3', { textContent: heading }), el('ol', {}, ...rows.map(item))))
  const skipped = letter.not_checked?.items || []
  $('#letter-body').replaceChildren(fields, ...(hidden ? [el('p', { className: 'notice plain noprint', textContent: EDITS_HIDDEN })] : []), ...(note ? [el('p', { className: 'note', textContent: note })] : []),
    ...(kept.length ? [el('p', { textContent: 'Please correct the following.' })] : []), ...sheets,
    ...(skipped.length ? [el('section', {}, el('h3', { textContent: letter.not_checked.line }), el('ul', {}, ...skipped.map(line => el('li', { textContent: line }))))] : []))
}

function renderSecond() {
  const { recheck, parent, sheets } = state.review
  const body = $('#second-body')
  if (!parent) return body.replaceChildren(el('p', { className: 'note', textContent: 'This is the first review of this set. Upload the resubmittal below to compare it with this review.' }))
  const link = el('p', {}, el('a', { href: `index.html?review=${parent}`, textContent: 'Open the earlier review' }))
  if (!recheck) return body.replaceChildren(link, el('p', { className: 'note', textContent: 'The comparison with the earlier review did not finish. See Not checked.' }))
  const sheetName = id => nameOf(sheets.find(s => s.id === id))
  const crop = (url, label, id) => el('figure', {}, url ? el('img', { src: url, alt: `${label} drawing at comment ${id}` }) : el('div', { className: 'nocrop', textContent: 'No crop' }), el('figcaption', { textContent: label }))
  const comments = recheck.comments.map(c => el('article', { className: 'compare' },
    el('div', { className: 'itemhead' }, el('span', { className: 'id', textContent: c.comment_id }), pill(RECHECK[c.state], `recheck ${c.state}`)),
    el('h4', { textContent: c.title }), el('p', { className: 'note', textContent: c.why }),
    ...(c.old_crop || c.new_crop ? [el('div', { className: 'pair' }, crop(c.old_crop, 'Earlier', c.comment_id), crop(c.new_crop, 'Resubmittal', c.comment_id))] : [])))
  const summary = ['resolved_looks', 'open', 'unknown'].map(name => `${RECHECK[name]}: ${recheck.comments.filter(c => c.state === name).length}`).join('  ·  ')
  const changes = recheck.changes.map(c => el('li', {}, el('button', { className: 'row', onclick: () => { $('#second').close(); focusItem(c) } },
    el('span', { className: 'title' }, ...(c.state ? [pill(RECHECK[c.state], `recheck ${c.state}`)] : []), ` ${c.note}`), el('span', { className: 'facts', textContent: sheetName(c.sheet) }))))
  const clouds = recheck.clouds.map(c => el('li', { textContent: `${sheetName(c.sheet)}${c.delta ? `, delta ${c.delta}` : ', no delta read'}` }))
  const unmatched = recheck.unmatched_sheets.map(name => el('li', { textContent: name }))
  body.replaceChildren(link, el('p', { className: 'summary', textContent: summary }), ...(comments.length ? comments : [el('p', { className: 'note', textContent: 'No comment was compared. See Not checked.' })]),
    el('h3', { className: 'pane-title', textContent: 'Changes no comment mentions' }), changes.length ? el('ul', { className: 'rows' }, ...changes) : el('p', { className: 'note', textContent: 'None were found.' }),
    el('h3', { className: 'pane-title', textContent: 'Sheets that could not be matched' }), unmatched.length ? el('ul', { className: 'lines' }, ...unmatched) : el('p', { className: 'note', textContent: 'Every sheet was matched by number.' }),
    el('h3', { className: 'pane-title', textContent: 'Revision clouds listed by the model' }), el('p', { className: 'note', textContent: 'A draft list. An empty list does not mean nothing changed.' }), ...(clouds.length ? [el('ul', { className: 'lines' }, ...clouds)] : []))
}

function renderAll() {
  renderMeta(); renderFloors(); renderMatrix(); renderCell(); renderFindings(); renderSet(); renderOpen(); renderTabs(); renderSheet()
}

function show(review) {
  state.review = review
  const { findings, project, floor_order, sheets } = review
  state.sheet = findings.find(f => f.severity === 'fix' && f.sheet != null)?.sheet ?? project.discharge_level ?? floor_order[0] ?? sheets[0]?.id ?? null
  document.body.dataset.view = 'review'
  $('#play').hidden = floor_order.length < 2
  floor_order.forEach(id => layerOf(sheetOf(id)))
  renderAll()
  fit()
}

async function choose() {
  say('')
  try {
    if (SAMPLE) {
      const decisions = state.review.decisions
      state.review = await loadSample($('#jurisdiction').value)
      state.review.decisions = decisions
    } else {
      state.review = await post('evaluate', choices())
    }
    renderAll()
  } catch (e) {
    say(e.message)
    renderMeta()
  }
}

function choices() {
  const chosen = $('#discharge').value
  return { jurisdiction: $('#jurisdiction').value, sprinklers: $('#sprinklers').value || null, discharge: chosen === '' || chosen === String(state.labelled) ? null : Number(chosen) }   // a discharge left as labelled stays out of the choices, so the model's letter wording can come back
}

function wire() {
  const v = $('#viewer')
  let drag = null
  v.addEventListener('pointerdown', e => { if (e.target.closest('.tools')) return; drag = { x: e.clientX - state.x, y: e.clientY - state.y }; v.setPointerCapture(e.pointerId); v.classList.add('dragging') })
  v.addEventListener('pointermove', e => { if (!drag) return; state.x = e.clientX - drag.x; state.y = e.clientY - drag.y; applyView() })
  v.addEventListener('pointerup', () => { drag = null; v.classList.remove('dragging') })
  v.addEventListener('wheel', e => { e.preventDefault(); const r = v.getBoundingClientRect(); zoomAt(e.clientX - r.left, e.clientY - r.top, Math.exp(-e.deltaY * .0016)) }, { passive: false })
  $('#zoom-in').onclick = () => { const r = v.getBoundingClientRect(); zoomAt(r.width / 2, r.height / 2, 1.3) }
  $('#zoom-out').onclick = () => { const r = v.getBoundingClientRect(); zoomAt(r.width / 2, r.height / 2, 1 / 1.3) }
  $('#fit').onclick = fit
  $('#play').onclick = toggleWalk
  $('#toggle-stairs').onclick = e => { state.showStairs = !state.showStairs; e.currentTarget.setAttribute('aria-pressed', state.showStairs); renderSheet() }
  $('#toggle-findings').onclick = e => { state.showFindings = !state.showFindings; e.currentTarget.setAttribute('aria-pressed', state.showFindings); renderSheet() }
  $('#open-letter').onclick = () => { renderLetter(); $('#letter').showModal(); $('#letter').scrollTop = 0 }   // the top of the letter says when an edit is hidden
  $('#close-letter').onclick = () => $('#letter').close()
  $('#print-letter').onclick = () => print()
  // A modal stays in the browser's top layer, where a long letter cannot run on to a second printed page.
  addEventListener('beforeprint', () => { if ($('#letter').matches(':modal')) { $('#letter').close(); $('#letter').show() } })
  addEventListener('afterprint', () => { if ($('#letter').open) { $('#letter').close(); $('#letter').showModal() } })
  $('#open-second').onclick = () => { $('#second').showModal(); renderSecond() }
  $('#close-second').onclick = () => $('#second').close()
  for (const id of ['#jurisdiction', '#sprinklers', '#discharge']) $(id).onchange = choose
  document.querySelectorAll('[role="tab"]').forEach((tab, i, tabs) => {
    tab.onclick = () => selectTab(tab.id.slice(4))
    tab.onkeydown = e => {
      const d = { ArrowRight: 1, ArrowLeft: -1 }[e.key]
      if (d) selectTab(tabs[(i + d + tabs.length) % tabs.length].id.slice(4), true)
    }
  })
  selectTab('findings')
  addEventListener('keydown', e => {
    if (e.target.closest('button, select, input, dialog, [contenteditable]')) return
    if (e.key === 'ArrowUp') stepFloor(1)
    if (e.key === 'ArrowDown') stepFloor(-1)
  })
  new ResizeObserver(() => state.review && fit()).observe(v)
}

function stopped(message) {
  $('#progress-message').textContent = message
  $('#progress-message').hidden = !message
}

function cannotShow(message) {
  document.body.dataset.view = 'progress'
  $('#progress-step').textContent = 'This review cannot be shown'
  stopped(message)
}

const stageMessages = {}
function whyStageFailed(name) {
  return stageMessages[name] ??= request(`${BASE}stage_${name}.json`).then(stage => stage.failed || '', () => '')
}

function stageRow(row, ended) {
  const count = row.total ? `${row.done} of ${row.total}` : ''
  const bar = row.state === 'running' ? [el('progress', Object.assign(row.total ? { max: row.total, value: row.done } : {}, { ariaLabel: row.label }))] : []
  const tr = el('tr', { className: row.state }, el('th', { scope: 'row', textContent: row.label }), el('td', { className: 'word', textContent: ended && row.state === 'queued' ? 'Not run' : STAGE[row.state] }),
    el('td', { className: 'count', textContent: count }, ...bar))
  if (row.state === 'failed') whyStageFailed(row.name).then(why => { if (why) tr.append(el('td', { className: 'why', textContent: why })) })
  return tr
}

function showProgress(status) {
  const ended = status.state === 'failed'
  $('#progress-step').textContent = ended ? 'This review stopped' : status.step || STAGE.queued
  $('#progress-facts').textContent = [status.sheets && sheetCount(status.sheets), `${(status.usd || 0).toFixed(2)} USD${ended || status.state === 'done' ? '' : ' so far'}`].filter(Boolean).join('  ·  ')
  $('#stages').replaceChildren(...(status.stages || []).map(row => stageRow(row, ended)))
  $('.schedule').hidden = !status.stages?.length
  stopped(ended ? status.message || 'The review stopped before it finished.' : '')
  const read = (status.stages || []).find(row => row.name === 'read')
  const seen = Math.min(status.sheets || 0, read?.state === 'done' ? Infinity : read?.done || 0)
  const thumbs = $('#thumbs')
  // shortcut: thumbnails are the full sheet PNGs, about 2 MB each, because no small image exists before assemble, add a thumbnail in ingest if a large set drags here.
  for (let i = thumbs.children.length; i < seen; i++) thumbs.append(el('img', { src: `${BASE}sheet_${String(i).padStart(3, '0')}.png`, alt: `Sheet ${i + 1}`, loading: 'lazy' }))
  $('#thumbs').parentElement.hidden = !seen
}

async function showName() {
  if (!$('#project').textContent) $('#project').textContent = (await request(BASE + 'stage_upload.json').catch(() => ({}))).name || ''
}

async function openReview(status) {
  try {
    const review = await request(BASE + 'review.json', { cache: 'no-cache' })   // every Keep and selector rewrites this file, and with no Cache-Control a browser reuses its old copy
    await loadRegistration(review)
    show(review)
  } catch (e) {
    showProgress(status)
    cannotShow(e.status === 401 ? LOCKED : e.status === 404 ? 'This review finished with nothing to show, because its last stage did not finish.' : 'This review could not be opened.')
    await showName()
  }
}

async function follow() {
  let status
  try {
    status = await request(`/api/reviews/${REVIEW}`)
  } catch (e) {
    if (e.status === 401 || e.status === 404) return cannotShow(e.status === 401 ? LOCKED : 'There is no such review.')
    stopped('Lost contact with the server. Trying again.')
    return setTimeout(follow, POLL)
  }
  state.status = status
  if (status.state === 'done') return openReview(status)
  await showName()
  showProgress(status)
  if (status.state !== 'failed') setTimeout(follow, POLL)
}

async function loadSample(jurisdiction) {
  const r = await fetch(`${BASE}review.${jurisdiction}.json`)
  if (!r.ok) throw new Error(`Could not load the ${jurisdiction} review.`)
  return JSON.parse((await r.text()).replace(/\/api\/reviews\/[0-9a-f]{32}\/files\//g, BASE))
}

async function openSample() {
  const entry = (await samples()).find(s => s.id === SAMPLE)
  if (!entry) return cannotShow('There is no such sample review.')
  document.querySelectorAll('#jurisdiction option').forEach(o => { if (!entry.jurisdictions.includes(o.value)) o.remove() })
  try {
    const review = await loadSample(entry.jurisdictions[0])
    review.decisions = remembered.get(`storey.decisions.${SAMPLE}`, {})
    await loadRegistration(review)
    show(review)
  } catch (e) {
    cannotShow(e.message)
  }
}

async function start() {
  if (!REVIEW && !SAMPLE) return location.replace('start.html' + location.hash)
  await openSession()
  document.body.dataset.mode = REVIEW ? 'live' : 'sample'
  wire()
  if (REVIEW) wireUpload($('#resubmit'), `/api/reviews/${REVIEW}/recheck`)
  return REVIEW ? follow() : openSample()
}
start()
