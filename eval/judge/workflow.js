export const meta = {
  name: 'storey-judge-claims',
  description: 'Two blind judges map each claim on each match sheet to the truth items it states; an adjudicator rules on every disagreement.',
  phases: [
    { title: 'Judge', detail: 'two blind judges per sheet, then an adjudicator on disagreements' },
  ],
}

const RULES = `You judge one part of the evaluation of an AI plan review tool for city building departments. Each sheet file is JSON: "set" (the building drawing set), "claims" (problems someone reported about that set: id, sheet, statement) and "truth" (problems known to be in that set: id, sheets, statement, status).

For each claim, list the ids of the truth items it states: the same problem at the same place. An empty list means it states none.

Rules:
1. Same problem: the claim names the same defect as the truth item, in any wording. Example: truth "Door tag 2002 on the second floor plan has no row in the door schedule" and claim "Door 2002 appears on A1.2 but is missing from the schedule on A6.1" match.
2. Same place: the claim must point at the same sheet, floor, stair, door, room, table row or index entry as the truth item. The same kind of problem at another door, room, floor or sheet does not match.
3. A general claim with no specific place or no specific discrepancy (for example "verify stair continuity", "coordinate schedules with plans", "confirm egress") states nothing.
4. One claim may state several truth items if it names each of them. Several claims may state the same truth item.
5. Severity, confidence and recommended action do not matter.
6. A truth item "The stair X is erased on floor N" is stated by a claim that a stair is missing, not drawn, or does not continue on that floor at that stair's place. A floor may be named in words ("third floor") or by sheet.
7. The status of a truth item (real, likely or other) does not matter for matching. Map to it if the claim states it.
8. Use only the sheet file you are given. Do not open any other file, drawing or folder.
9. When unsure whether a claim states a truth item, do not match it, and say why in the note.

Return one ruling for every claim on each sheet, in the order the claims are listed, each with: id (the claim's id), truth (list of truth ids, possibly empty) and a short note giving the reason.`

const SCHEMA = {
  type: 'object',
  properties: { sheets: { type: 'array', items: { type: 'object', properties: {
    sheet: { type: 'string' },
    matches: { type: 'array', items: { type: 'object', properties: { id: { type: 'string' }, truth: { type: 'array', items: { type: 'string' } }, note: { type: 'string' } }, required: ['id', 'truth', 'note'] } },
  }, required: ['sheet', 'matches'] } } },
  required: ['sheets'],
}
const ADJ = {
  type: 'object',
  properties: { rulings: { type: 'array', items: { type: 'object', properties: { sheet: { type: 'string' }, id: { type: 'string' }, truth: { type: 'array', items: { type: 'string' } }, note: { type: 'string' } }, required: ['sheet', 'id', 'truth', 'note'] } } },
  required: ['rulings'],
}

const dir = args.dir
const heavy = new Set(args.heavy)
const light = args.names.filter(n => !heavy.has(n))
const batches = [...args.heavy.map(n => [n])]
for (let i = 0; i < light.length; i += 3) batches.push(light.slice(i, i + 3))

const key = r => JSON.stringify([...r.truth].sort())

const results = await pipeline(batches,
  batch => parallel([1, 2].map(k => () => agent(`${RULES}\n\nJudge ${batch.length === 1 ? 'this sheet' : 'these sheets'}: ${batch.map(n => `${dir}/${n}.json (sheet name ${n})`).join(', ')}. Return each sheet under its sheet name.`,
    { label: `judge${k}:${batch.join(',')}`, phase: 'Judge', schema: SCHEMA }))),
  async (pair, batch) => {
    const [a, b] = pair
    if (!a || !b) return { batch, error: 'a judge returned nothing' }
    const out = []
    const disputes = []
    for (const name of batch) {
      const ma = (a.sheets.find(s => s.sheet === name) || { matches: [] }).matches
      const mb = (b.sheets.find(s => s.sheet === name) || { matches: [] }).matches
      const byB = new Map(mb.map(m => [m.id, m]))
      const rows = []
      for (const m of ma) {
        const other = byB.get(m.id)
        if (other && key(other) === key(m)) rows.push({ id: m.id, truth: m.truth, note: m.note, agreed: true })
        else { rows.push({ id: m.id, truth: null, note: '', agreed: false }); disputes.push({ sheet: name, id: m.id, first: { truth: m.truth, note: m.note }, second: other ? { truth: other.truth, note: other.note } : null }) }
      }
      for (const m of mb) if (!ma.some(x => x.id === m.id)) { rows.push({ id: m.id, truth: null, note: '', agreed: false }); disputes.push({ sheet: name, id: m.id, first: null, second: { truth: m.truth, note: m.note } }) }
      out.push({ sheet: name, rows })
    }
    if (disputes.length) {
      const adj = await agent(`${RULES}\n\nYou are the adjudicator. Two judges, working blind to each other, disagreed on the claims below. Read the sheet files named, weigh both rulings against the rules, and give the final ruling for each listed claim only.\n\nSheet files: ${batch.map(n => `${dir}/${n}.json (sheet name ${n})`).join(', ')}\n\nDisagreements:\n${JSON.stringify(disputes, null, 1)}`,
        { label: `adjudicate:${batch.join(',')}`, phase: 'Judge', schema: ADJ })
      for (const r of (adj && adj.rulings) || []) {
        const sheet = out.find(s => s.sheet === r.sheet)
        const row = sheet && sheet.rows.find(x => x.id === r.id)
        if (row) { row.truth = r.truth; row.note = `Adjudicated: ${r.note}` }
      }
    }
    return { batch, sheets: out, disputes: disputes.length }
  })
return results
