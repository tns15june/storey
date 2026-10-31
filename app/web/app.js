// Review screen. Reads one review JSON and draws it. All state lives here so a new review only needs new JSON.
const SEVERITY = { fix: 'Must fix', check: 'Check', unknown: 'Cannot tell' }
const COLORS = ['#1f8a4c', '#1f6fd6', '#a42bb8', '#d9730d', '#0f8a94', '#6d5bd0', '#b5483a', '#3b7d23']
const REVIEW_ID = new URLSearchParams(location.search).get('review')
const REVIEW_URL = /^[0-9a-f]{32}$/.test(REVIEW_ID || '') ? `/api/reviews/${REVIEW_ID}/files/review.json` : 'sample/review.json'
const $ = s => document.querySelector(s)
const el = (tag, props = {}, ...kids) => { const e = Object.assign(document.createElement(tag), props); e.append(...kids); return e }
const NS = 'http://www.w3.org/2000/svg'
const svgEl = (tag, attrs) => { const e = document.createElementNS(NS, tag); for (const k in attrs) e.setAttribute(k, attrs[k]); return e }

const state = { review: null, sheet: null, active: null, decisions: {}, showStairs: true, showFindings: true, scale: 1, x: 0, y: 0, playing: null }

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

function renderMeta() {
  const p = state.review.project
  $('#project').textContent = p.name
  $('#address').textContent = p.address
  $('#cycle').textContent = `Cycle ${p.cycle}`
  $('#code').textContent = p.code
  $('#note').textContent = p.note || ''
}

function renderFloors() {
  const box = $('#floors')
  box.querySelectorAll('.floor').forEach(n => n.remove())
  for (const s of state.review.sheets) {
    const n = state.review.findings.filter(f => f.sheet === s.id && f.severity !== 'unknown').length
    const b = el('button', { className: 'floor', onclick: () => go(s.id) },
      el('img', { src: s.image, alt: '', loading: 'lazy' }), el('span', { className: 'name', textContent: s.level }))
    b.dataset.id = s.id
    if (n) b.append(el('span', { className: 'count', textContent: n, title: `${n} on this floor` }))
    box.append(b)
  }
}

function renderMatrix() {
  const { sheets, verticals, findings } = state.review
  const head = el('tr', {}, el('th', { scope: 'col' }))
  verticals.forEach(v => head.append(el('th', { scope: 'col', textContent: v.id })))
  const rows = sheets.map(s => {
    const tr = el('tr', {}, el('th', { scope: 'row', className: 'row', textContent: s.level }))
    tr.dataset.id = s.id
    verticals.forEach(v => {
      const present = !!v.floors[s.id], gap = findings.some(f => f.vertical === v.id && f.sheet === s.id && f.severity === 'fix')
      const btn = el('button', { title: `Stair ${v.id}, ${s.level}: ${gap ? 'gap' : present ? 'found' : 'none'}`, onclick: () => go(s.id) })
      btn.setAttribute('aria-label', btn.title)
      tr.append(el('td', { className: gap ? 'gap' : present ? 'yes' : 'no' }, btn))
    })
    return tr
  })
  $('#matrix').replaceChildren(el('thead', {}, head), el('tbody', {}, ...rows))
}

function renderFindings() {
  const list = $('#findings')
  list.replaceChildren(...state.review.findings.map(f => {
    const where = state.review.sheets.find(s => s.id === f.sheet)
    const delta = svgEl('svg', { class: 'delta', viewBox: '0 0 30 28', 'aria-hidden': 'true' })
    delta.append(svgEl('path', { d: 'M15 2 L28 26 H2 Z' }))
    const t = svgEl('text', { x: 15, y: 22 }); t.textContent = f.id; delta.append(t)
    const head = el('button', { className: 'head', onclick: () => focus(f) }, delta,
      el('div', {}, el('span', { className: 'pill', textContent: SEVERITY[f.severity] }), el('h3', { textContent: f.title }),
        el('div', { className: 'where', textContent: `${where ? where.level : ''}  ·  ${f.rule}` })))
    const dl = el('dl', {})
    for (const [k, v] of [['Found', f.measured], ['Needed', f.required], ['Why we think so', f.evidence]]) if (v) dl.append(el('dt', { textContent: k }), el('dd', { textContent: v }))
    const decide = el('div', { className: 'decide' })
    for (const [key, label] of [['keep', 'Keep'], ['drop', 'Drop']]) {
      const b = el('button', { textContent: label, onclick: () => { state.decisions[f.id] = state.decisions[f.id] === key ? null : key; renderFindings() } })
      b.setAttribute('aria-pressed', state.decisions[f.id] === key)
      decide.append(b)
    }
    const card = el('article', { className: `finding ${f.severity}${state.decisions[f.id] === 'drop' ? ' dropped' : ''}` }, head, dl, decide)
    card.setAttribute('aria-current', state.active === f.id)
    return card
  }))
}

function renderSheet() {
  const s = state.review.sheets.find(x => x.id === state.sheet)
  const img = $('#sheet')
  if (img.getAttribute('src') !== s.image) img.src = s.image
  img.width = s.w; img.height = s.h
  const stage = $('#stage'); stage.style.width = `${s.w}px`; stage.style.height = `${s.h}px`
  const svg = $('#overlay'); svg.setAttribute('viewBox', `0 0 ${s.w} ${s.h}`); svg.replaceChildren()
  const stroke = 4 / Math.min(1, state.scale * 1.6)
  if (state.showStairs) state.review.verticals.forEach((v, i) => {
    const f = v.floors[s.id]; if (!f) return
    const [x1, y1, x2, y2] = f.box
    svg.append(svgEl('rect', { x: x1 - 5, y: y1 - 5, width: x2 - x1 + 10, height: y2 - y1 + 10, fill: 'none', stroke: COLORS[i % COLORS.length], 'stroke-width': stroke }))
    const t = svgEl('text', { x: x1 - 4, y: y1 - 12, fill: COLORS[i % COLORS.length], 'font-size': 26, 'font-family': 'IBM Plex Sans Condensed, sans-serif', 'font-weight': 700 })
    t.textContent = `${v.id}  ${Math.round(f.conf * 100)}%`; svg.append(t)
  })
  if (state.showFindings) state.review.findings.filter(f => f.sheet === s.id && f.box).forEach(f => {
    const color = getComputedStyle(document.documentElement).getPropertyValue(f.severity === 'fix' ? '--fix' : '--check')
    const [x1, y1, x2, y2] = f.box
    svg.append(svgEl('path', { d: cloudPath([x1 - 14, y1 - 14, x2 + 14, y2 + 14], 13), fill: state.active === f.id ? color : 'none', 'fill-opacity': .12, stroke: color, 'stroke-width': stroke * 1.2 }))
    const tx = x2 + 34, ty = y1 - 10
    svg.append(svgEl('path', { d: `M ${tx} ${ty - 26} L ${tx + 28} ${ty + 22} H ${tx - 28} Z`, fill: '#fff', stroke: color, 'stroke-width': stroke }))
    const t = svgEl('text', { x: tx, y: ty + 15, 'text-anchor': 'middle', fill: color, 'font-size': 26, 'font-weight': 700, 'font-family': 'IBM Plex Mono, monospace' })
    t.textContent = f.id; svg.append(t)
  })
  $('#floorname').textContent = `${s.level}  ·  ${s.title}`
  document.querySelectorAll('.floor').forEach(b => b.setAttribute('aria-current', b.dataset.id === s.id))
  document.querySelectorAll('#matrix tbody tr').forEach(r => r.classList.toggle('here', r.dataset.id === s.id))
}

function applyView(ease) {
  const stage = $('#stage')
  stage.classList.toggle('ease', !!ease)
  stage.style.transform = `translate(${state.x}px, ${state.y}px) scale(${state.scale})`
}
function fit() {
  const s = state.review.sheets.find(x => x.id === state.sheet), v = $('#viewer').getBoundingClientRect()
  state.scale = Math.min(v.width / s.w, v.height / s.h) * .97
  state.x = (v.width - s.w * state.scale) / 2; state.y = (v.height - s.h * state.scale) / 2
  applyView(); renderSheet()
}
function zoomAt(cx, cy, k) {
  const next = Math.min(4, Math.max(.1, state.scale * k)), r = next / state.scale
  state.x = cx - (cx - state.x) * r; state.y = cy - (cy - state.y) * r; state.scale = next
  applyView(); renderSheet()
}
function go(id) { state.sheet = id; renderSheet() }
function focus(f) {
  state.active = f.id
  if (f.sheet) go(f.sheet)
  renderFindings(); renderSheet()
  if (f.box) {
    const v = $('#viewer').getBoundingClientRect(), [x1, y1, x2, y2] = f.box
    state.scale = Math.min(1.2, Math.min(v.width, v.height) / (Math.max(x2 - x1, y2 - y1) * 3.2))
    state.x = v.width / 2 - (x1 + x2) / 2 * state.scale; state.y = v.height / 2 - (y1 + y2) / 2 * state.scale
    applyView(true); renderSheet()
  }
}
function step(d) {
  const ids = state.review.sheets.map(s => s.id), i = ids.indexOf(state.sheet)
  go(ids[(i + d + ids.length) % ids.length])
}
function togglePlay() {
  const b = $('#play')
  if (state.playing) { clearInterval(state.playing); state.playing = null; b.setAttribute('aria-pressed', false); b.textContent = 'Walk the floors'; return }
  b.setAttribute('aria-pressed', true); b.textContent = 'Stop'
  state.playing = setInterval(() => step(1), 1100)
}

function openLetter() {
  const { project, findings, sheets } = state.review
  const kept = findings.filter(f => f.severity !== 'unknown' && state.decisions[f.id] !== 'drop')
  $('#letter-body').replaceChildren(
    el('p', { textContent: `${project.name}. Plan review, cycle ${project.cycle}. Code: ${project.code}.` }),
    el('p', { textContent: 'Please correct the following and resubmit.' }),
    el('ol', {}, ...kept.map(f => el('li', { textContent: `${sheets.find(s => s.id === f.sheet).level}: ${f.title}. ${f.measured} ${f.required} (${f.rule})` }))),
    el('p', { className: 'note', textContent: 'Draft for the examiner. A person reviews and signs this letter.' }))
  $('#letter').hidden = false
  $('#close-letter').focus()
}

function wireViewer() {
  const v = $('#viewer')
  let drag = null
  v.addEventListener('pointerdown', e => { if (e.target.closest('.tools')) return; drag = { x: e.clientX - state.x, y: e.clientY - state.y }; v.setPointerCapture(e.pointerId); v.classList.add('dragging') })
  v.addEventListener('pointermove', e => { if (!drag) return; state.x = e.clientX - drag.x; state.y = e.clientY - drag.y; applyView() })
  v.addEventListener('pointerup', () => { drag = null; v.classList.remove('dragging') })
  v.addEventListener('wheel', e => { e.preventDefault(); const r = v.getBoundingClientRect(); zoomAt(e.clientX - r.left, e.clientY - r.top, Math.exp(-e.deltaY * .0016)) }, { passive: false })
  $('#zoom-in').onclick = () => { const r = v.getBoundingClientRect(); zoomAt(r.width / 2, r.height / 2, 1.3) }
  $('#zoom-out').onclick = () => { const r = v.getBoundingClientRect(); zoomAt(r.width / 2, r.height / 2, 1 / 1.3) }
  $('#fit').onclick = fit
  $('#play').onclick = togglePlay
  $('#toggle-stairs').onclick = e => { state.showStairs = !state.showStairs; e.currentTarget.setAttribute('aria-pressed', state.showStairs); renderSheet() }
  $('#toggle-findings').onclick = e => { state.showFindings = !state.showFindings; e.currentTarget.setAttribute('aria-pressed', state.showFindings); renderSheet() }
  $('#open-letter').onclick = openLetter
  $('#close-letter').onclick = () => { $('#letter').hidden = true }
  addEventListener('keydown', e => {
    if (e.key === 'Escape') $('#letter').hidden = true
    if (e.target.closest('button')) return
    if (e.key === 'ArrowUp') step(-1)
    if (e.key === 'ArrowDown') step(1)
  })
  new ResizeObserver(() => state.review && fit()).observe(v)
}

async function start() {
  const r = await fetch(REVIEW_URL)
  if (!r.ok) throw new Error(`Could not load ${REVIEW_URL}: ${r.status}`)
  state.review = await r.json()
  state.sheet = (state.review.findings.find(f => f.severity === 'fix') || {}).sheet || state.review.sheets[0].id
  renderMeta(); renderFloors(); renderMatrix(); renderFindings(); wireViewer(); fit()
}
start().catch(e => { $('#note').textContent = e.message })
