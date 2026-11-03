# Reproducing the evaluation

The published numbers come from one run of 33 sets on 1 November 2026, on the code, scoring and truth that `eval/frozen.json` pins by sha256. This file lists every input set, what this repository does not hold and why, and the commands that remake the inputs, rerun the sets, judge the claims and write the report.

A rerun calls the models again, so its claims and numbers can differ from the published ones, and its claims need judging again. The 33 runs of 1 November cost 16.23 USD for Storey and 2.35 USD for the baseline (`docs/research/build-log.md`).

What can be reproduced:

- The 29 development sets (22 generated, 6 planted erasures, `mml_capitol`) can be rerun from their public sources.
- The three held-out sets (`army_bt_bcof`, `army_bt_bcof_plant`, `queens_royce`) cannot be run again as held out, on any machine. That is deliberate: a held-out set runs once, so its number cannot be the best of several. See [eval/frozen.json](#evalfrozenjson).
- `va_trb_richmond` cannot be scored by anyone else, because its truth is not distributed.
- Anyone can check that a checkout's `storey/`, scoring files and truth files hash to the values in `eval/frozen.json`.

## Input sets

`data/` is not in git, so every file below has to be fetched or made again. Paths are under `data/`.

### Generated sets (22, development)

Each is drawn by `eval.generate` from one building of Swiss Dwellings v3.0.0 (<https://zenodo.org/records/7788422>, CC BY 4.0), saved as `plans/swiss_dwellings/swiss-dwellings-v3.0.0.zip`, with seed 1. It writes `generated/<set>/set.pdf` and `truth.json`, and the truth pins `set.pdf` by `pdf_sha256`. A `_second` set is the resubmittal of its `_mixed` set: the driver starts it after its first, and both share the first's model cache. On the author's machine, with the versions in `requirements.txt`, all 22 sets made again gave the same `set.pdf`, byte for byte, and the same truth. These sets are shown in full on the How well it works page.

| Set | Building | Variant | `set.pdf` sha256 |
|---|---|---|---|
| `710_1_clean` | 710 | clean | `0e9a638c805cdbf01866f53ffc4d23ea7b7339d01c59c619ac664b789fab895c` |
| `710_1_mixed` | 710 | mixed | `67b00b403fee787967de10c163689f8e0d3d25c0d6adaca38e709cd3620dce7d` |
| `710_1_second` | 710 | second | `ba0132239b71bfd794582393ed5e85dd9aed6d07d56ac9c95874c022911fc126` |
| `3370_1_clean` | 3370 | clean | `643dba2fcc95f0da69b8cdc2c2c815eda0875c3d3e297b879b70f242baef77bf` |
| `3370_1_mixed` | 3370 | mixed | `4f9ae257e1730d337260abc791d9f97dfe8cd8c234322b1471c50b7b51fe0afa` |
| `3370_1_second` | 3370 | second | `51dea4d3c371284efdc9a24cec806f4d9c85a716a6356e6127eadc38bc7f1fde` |
| `5615_1_clean` | 5615 | clean | `40542d9f38eb8471fe3203e3bd0773c7f7f14ec8a7f66da91dd55e1ea96cd2ca` |
| `5615_1_mixed` | 5615 | mixed | `238e19149da79a16c6c741fb399873287bdc6a8f68fe17546dd832a4a25f9662` |
| `5615_1_second` | 5615 | second | `7b5b7ab31b2a289151db4edf59b1d524a273acdfc9706df53f0fcc96b13f1001` |
| `16281_1_clean` | 16281 | clean | `1f56a6a57a30b0e611e299c6da7ec5eeb7010dde11d375a2675d80ac6d833b6e` |
| `16281_1_mixed` | 16281 | mixed | `40f78d613f0bfcacc1bb53f2b98b64b3ea644abff09038d864d5732f9f2aeaf9` |
| `16281_1_second` | 16281 | second | `ffc393bf3ee8b3bd5db812423e7d16d4552d35e27bbc3e2c37b6b29b4a618602` |
| `3128_1_clean` | 3128 | clean | `16188bf039040c77802e7c776cb4a398f431bafaf4420ea6d3155c29014af208` |
| `3128_1_mixed` | 3128 | mixed | `5b20cf1f4161568489043f72ed67f9977e7858d1756c1c5670a03a2f6333d52d` |
| `3128_1_second` | 3128 | second | `2eb25a210398d6bfb014d846c5d7060ae22802d9dc15ba2f9f24123d0e6bc373` |
| `2828_1_clean` | 2828 | clean | `9ecfec625d9d861e246120503a0c2039e3ea3f5901d7e2cdb0ef77c233451a0c` |
| `2828_1_mixed` | 2828 | mixed | `262924ce11eb718d98873da245db1d90e198e34a833fb39a3e005f5b91652262` |
| `2828_1_second` | 2828 | second | `3029174ed5a44e2bfd18f92cbaf95b5afde4b86776decf7e4b68121ceb07d21d` |
| `2828_1_corridor` | 2828 | corridor | `b3c5a85a6a038d350e62f525762ddcbda20a61ccd04a4d68131bd57c1b49622c` |
| `6267_1_clean` | 6267 | clean | `201fe8ea055d5190d38502b55f3c12f9c8d2da3d4254f5eadee6f8423f350b55` |
| `6267_1_mixed` | 6267 | mixed | `10bc636ae14c1cf10be526a6933508fd3afe023156dc46afb0c6c8b054eb990a` |
| `6267_1_second` | 6267 | second | `180d63f10efcd56547709962eb482734764b023d09fc83bd4695d29b18be44b6` |

### Planted erasures on real sheets (6, development)

`eval.plant <building>` erases stairs on copies of a building's floor plans and writes `eval/plants/<building>_plant/`. The copies are the 2000 px renders `lab/presence/render.py` makes in `lab/presence/sheets/<building>/<n>.jpg`, numbered from 0 in the floor order below. The erased cells and every stair's boxes come from `lab/presence/truth.json` in this repository (sha256 `438b5738ee7b2893849d83913a89fcf5ab512128d5e5d49889ad4f599218f8fa`), which each plant's `truth.json` records as `from_sha256`. No truth file pins these six plants' images.

| Set | Building and source | Files under `plans/<building>/`, in floor order | Licence, from the catalog | On the page |
|---|---|---|---|---|
| `habs_stelizabeths_center_plant` | St. Elizabeths Hospital, Center Building: HABS DC-349-W, sheets 1 to 5 of 10, on Wikimedia Commons, saved as the 3840 px thumbnails | `00.jpg` (basement), `02.jpg`, `03.jpg`, `04.jpg`, `01.jpg` (sheet 5, drawn as the fourth floor) | Public domain, US federal record | lists and counts |
| `habs_stelizabeths_atkins_plant` | St. Elizabeths Hospital, Atkins Hall: HABS DC-349-T, sheets 1 to 5 of 11, Commons, 3840 px thumbnails | `01.jpg` (basement), `02.jpg`, `03.jpg`, `04.jpg`, `00.jpg` (attic) | Public domain, US federal record | lists and counts |
| `habs_va_bldg1_plant` | VA Medical Center Building No. 1, Huntington WV: HABS WVA,50-HUNT,1A-25 to 1A-29, Commons, 3840 px thumbnails | `02.jpg` (ground), `00.jpg`, `03.jpg`, `04.jpg`, `01.jpg` (fourth) | Public domain, US federal record | lists and counts |
| `schenley_plant` | Schenley High School, Pittsburgh, 1916: the four floor plan PNGs on Commons, unscaled | `sch_base.png`, `sch_ground.png`, `sch_first.png`, `sch_second.png` | PD-US, published 1916 | lists and counts |
| `wfu_huffman_plant` | Huffman Residence Hall, Wake Forest University: <https://prod.wp.cdn.aws.wfu.edu/sites/303/2017/07/HUFFMAN-FLOOR-PLAN-G-3.pdf> | `HUFFMAN-FLOOR-PLAN-G-3.pdf`, pages 1 to 4 | None stated, university copyright assumed | counts only |
| `bme_f_plant` | BME Building F, Budapest: the four level plans on Commons, 3840 px thumbnails | `bme_-1.jpg`, `bme_0.jpg`, `bme_1.jpg`, `bme_2.jpg` | PD-anon-expired on Commons, but the date tag cannot be trusted | counts only |

`bme_k_plant` was made but not run: its truth boxes are tighter than two of its three stairs, so those erasures left most of the treads.

### Real sets (5: one development, one examiner record, three held out)

| Set | Source | File under `plans/` and pages sent | Truth | Pinned sha256 | On the page |
|---|---|---|---|---|---|
| `mml_capitol` | The Capitol, Michigan Municipal League pattern book duplex: <https://mml.org/resources-research/books-and-briefs/pattern-book-homes-for-21st-century-michigan/volume-ii-the-missing-middle-mixtape/>. The publisher calls the sheets copyright-free construction documents, with no named licence | `mml_capitol/THE-CAPITOL.pdf`, every page | `eval/truth/mml_capitol.json`, a blind audit | PDF `3527108acfc08ee87fadaace106de10500207cb8020958d92b85e0a22996caa9` | lists and counts |
| `va_trb_richmond` | Virginia Technical Review Board, Appeal 26-02, in the board packet <https://www.dhcd.virginia.gov/sites/default/files/DocX/sbctrb/agendas/agendas-trb-agenda-06-12-2026.pdf> | `va_trb/agendas-trb-agenda-06-12-2026.pdf`, page 107 only. `eval.run` refuses pages 84, 86, 87, 101 and 176, which hold the examiner's and the appellant's marks | `lab/realsets/va_trb/richmond_truth.json`, from the examiner's letter. Not distributed | none published | counts only, never its lists |
| `army_bt_bcof` (held out) | US Army Corps of Engineers Fort Worth District, BT BCOF sample set: <https://mrsi.erdc.dren.mil/content/cos/swf/bt-osut/Library/Standard%20Designs/BT%20BCOF%20-%20sample%20set.pdf>. No licence statement; probably in-house federal work | `army_cos/bt_bcof_sample.pdf`, all 6 pages | `eval/truth/army_bt_bcof.json` (blind audit) and `eval/truth/army_bt_bcof_stairs.json` (stair truth) | PDF `0307ef8ad579b3df02092cb8606550670a36f3380cd7d0a4ea1b40feeff8c8b7`, in both | lists and counts |
| `army_bt_bcof_plant` (held out) | `eval.plant army_bt_bcof`: the six pages drawn as ingest draws a PDF, with the cells the stair truth picks erased | `eval/plants/army_bt_bcof_plant/000.png` to `005.png` | its `truth.json`, made from the stair truth (`from_sha256` `76c9763d0c5edcff32d79a6c20b793f42c178c0f2d3d76a2c0fbbdbb48f9ba8d`) | each image, below | lists and counts |
| `queens_royce` (held out) | Jean Royce Hall, Queen's University: <https://www.queensu.ca/residences/sites/residencewww/files/2025-06/Jean%20Royce%20Hall%20Floor%20Plans%2005.2025.pdf>. No licence stated, university copyright assumed | `queens_royce/jean_royce_hall.pdf`, pages 6 to 9 (Phase 2) | `eval/truth/queens_royce_stairs.json`, stair truth only | PDF `5b928b17af7e43644a652b4d52e880809080914b8602904d5fd3faca34be5ac4` | counts only |

`queens_royce` has no plant: the erasure, made for black lines on paper, left a flight of one stair on its colour-filled raster and put white patches over the other cells, so `eval.plant` refuses to write one.

The images of `army_bt_bcof_plant`, as its truth pins them:

| Image | sha256 |
|---|---|
| `000.png` | `c6e11ec08b091be524b0defe151b7d2071703986111a87c55bd7409415db0381` |
| `001.png` | `c95351d46c2801b68eac8523c27788ca5e9752aa36cdce5a892b17daa1cdff47` |
| `002.png` | `4d64b402fb44c680b171db388aabc01432a5e281b95af4f42c09c10c8630070f` |
| `003.png` | `4878ba5ba93e5148d0a038cfabfb138f7671324fee8bcf6a8c55b3fa928373fc` |
| `004.png` | `9805b3530a971e811330d6e0932a54cde09b6969db8ef99dd70614b56868ee0e` |
| `005.png` | `bdbb03aac4bef6d8337d761b5c9e09267b6c664acefb46f38673f03efed7a4e4` |

The truth hashes `eval/frozen.json` holds, one per set over all of the set's truth files:

| Set | Truth hash |
|---|---|
| `army_bt_bcof` | `f73dc5a28d749d18b541fc8d6cc2a93f75ed87ed08929bc421e58e8e151916fd` |
| `army_bt_bcof_plant` | `e5125d0dc179855851b4fc3113764652c283be5c0c0a417eaf7e2ba242706b4e` |
| `mml_capitol` | `0a249b53e3c9c7625e89a336e7234b2ca79ba41ecda4cc3b09a3ebb7ee1bce46` |
| `queens_royce` | `370d7c0a3ddd74bef0e2ac6d50bc99b513a8e2fd67136287a26a44e8c5106d42` |

## Not distributed, and why

- Nothing under `data/` is in git: the drawings, the generated sets, the plants, the review folders, the baseline records, the match sheets, the model cache and the catalogs (`data/plans/catalog.json` and `catalog_r2.json`), which hold the source, licence and floor list of each building. The tables above carry what the evaluation needs from them. Without the catalogs, `eval.report` shows counts alone for every set and `lab/presence/render.py` cannot run.
- Third-party drawings keep their own terms. The Wake Forest and Queen's drawings state no licence, so university copyright is assumed, and the Army sample states none either. The public domain scans and the MML sheets are not copied here either, because `data/` stays out of git as a whole; each is fetched from where it is published. How well it works shows a set's misses, false findings and wrong passes only when its catalog entry says `public_demo` yes; every other set shows counts alone.
- Swiss Dwellings is CC BY 4.0 and could be copied with credit, but it is more than 1 GB. Fetch it from Zenodo; the generated sets are made from it again.
- Richmond (`va_trb_richmond`) is never public. The packet is a public board record, but the sheet is the architect's work and carries tenant names, and its truth quotes the plans examiner's letter. `eval/report.py` keeps any set whose source starts with `va_trb` out of the page's lists whatever a catalog says, its truth stays in `data/lab/realsets`, and its judge mappings are git-ignored (`eval/matches/va_trb_richmond.*`).
- BCOE (`bcoe_center`, a Butte County Office of Education bid set) is never public: its sheets reserve the architect's copyright and forbid copying. `eval/report.py` keeps any source starting with `bcoe` off the page's lists. It is not one of the 33 sets.
- The review folders of the 1 November run hold renders of the drawings, so they are not distributed. They are on the author's machine in a temporary folder, the one `eval/frozen.json` names.

## Commands

All from the repository root, with Python 3.12.

### 1. The frozen product code

A held-out set runs only on the `storey/` that `eval/frozen.json` pins, which is the tree at commit `a45163c`, and later commits may change `storey/`. The scoring files and truth files have not changed since that commit. In a clone, put back that tree, and do not commit the result:

```sh
git clone https://github.com/tns15june/storey.git storey-eval && cd storey-eval
rm -rf storey && git checkout a45163c -- storey   # the product package, exactly as frozen
python3.12 -m venv .venv && .venv/bin/pip install -r requirements.txt
echo "MODEL_API_KEY=<your Meta Model API key>" > .env
.venv/bin/python -c "from eval import run; f = run.read_json(run.FROZEN); print(run.code_hash()[0] == f['storey'], run.eval_hash() == f['eval'], {n: run.truth_hash(run.truth_paths(n)) == h for n, h in f['truth'].items()})"
```

The last line prints `True True` and the truth checks. `army_bt_bcof_plant` prints `False` until its plant is made, and stays `False` in any clone but the author's, because its `truth.json` records the absolute path of the stair truth it was made from.

### 2. The inputs

Fetch each source into `data/plans/<id>/` under the file names in the tables. Then:

```sh
for b in 710 3370 5615 16281 3128 2828 6267; do
  for v in clean mixed second; do .venv/bin/python -m eval.generate --building $b --seed 1 --variant $v; done
done
.venv/bin/python -m eval.generate --building 2828 --seed 1 --variant corridor
shasum -a 256 data/generated/*/set.pdf          # against the table above

.venv/bin/python lab/presence/render.py
for b in habs_stelizabeths_center habs_stelizabeths_atkins habs_va_bldg1 schenley wfu_huffman bme_f; do .venv/bin/python -m eval.plant $b; done
.venv/bin/python -m eval.plant army_bt_bcof     # optional: its images can be checked against the table, and it cannot run again
```

`lab/presence/render.py` reads `data/plans/catalog.json` and renders every building in its `PICK` list, so it needs a catalog entry, with `floors` listing `file` and `page` in the order above, and the files for each building it lists, not only the six. Look at each plant's `check.jpg` before a run: a truth box drawn tighter than its stair leaves treads outside the box.

### 3. Rerun

Use a new data folder: `eval.run` refuses a folder that holds a finished run, and only a run begun in a new folder on a cold model cache has its cost and minutes counted. Name the 29 development sets; the three held-out sets would be refused, and Richmond cannot be scored without its truth.

```sh
.venv/bin/python -m eval.run_all <data folder> \
  710_1_clean 710_1_mixed 710_1_second 3370_1_clean 3370_1_mixed 3370_1_second 5615_1_clean 5615_1_mixed 5615_1_second \
  16281_1_clean 16281_1_mixed 16281_1_second 3128_1_clean 3128_1_mixed 3128_1_second 2828_1_clean 2828_1_mixed 2828_1_second \
  6267_1_clean 6267_1_mixed 6267_1_second 2828_1_corridor habs_stelizabeths_center_plant habs_stelizabeths_atkins_plant \
  habs_va_bldg1_plant schenley_plant wfu_huffman_plant bme_f_plant mml_capitol
```

The driver runs four sets at a time, each followed by the baseline, and starts a second review after its first. It writes one line per step to `<data folder>/progress.jsonl`. When every step is done it rebuilds `data/eval/runs.json` from the `eval_run.json` files under `<data folder>/reviews`, because `eval.run` writes that registry with no lock and two runs that end together can drop each other's entry. It exits naming any held-out run that `eval/frozen.json` does not record, and never writes that file. On 1 November it ran as `python eval/run_all.py <data folder>`, with no set list.

### 4. Judge

The mappings in `eval/matches/` answer the 1 November match sheets, and `eval.report` refuses a mapping whose `sheet_sha256` is not its sheet's. Move them out of the clone's `eval/matches/` before the first report on a rerun.

```sh
mkdir ../matches_1nov && mv eval/matches/*.json ../matches_1nov/
.venv/bin/python -m eval.report                                  # writes data/eval/sheets/, lists each sheet waiting for the judge, and stops
.venv/bin/python eval/judge/blind.py data/eval/sheets <blind folder>
```

In a Claude Code session, run `eval/judge/workflow.js` as a workflow with the arguments `{"dir": "<blind folder>", "names": [each sheet name in <blind folder>.key.json], "heavy": [names of sheets to judge one to a batch, or none: []]}`, and save the list it returns beside the key as `<blind folder>.rulings.json`. The key is written beside the blind folder, not in it, because the judges are given the folder. Then:

```sh
.venv/bin/python eval/judge/blind.py --unblind <blind folder> <blind folder>.rulings.json
```

writes `eval/matches/<set>.<method>.json` for each sheet the judges ruled on. [eval/judge/README.md](judge/README.md) has the rules and what blinding removes.

### 5. Report

```sh
.venv/bin/python -m eval.report
```

writes `data/eval/report.json` and `app/web/eval.json`. While any set waits for the judge or has no baseline scored, it prints what waits and leaves `app/web/eval.json` as it was. A rerun's numbers are development numbers, or post hoc where a set's truth changed after its run.

## eval/frozen.json

`eval/frozen.json` holds the sha256 of `storey/` (`storey`), of `eval/run.py`, `eval/score.py` and `eval/baseline.py` (`eval`), of each set's truth files (`truth`), the time of the freeze (`frozen`, 2026-11-01 23:06 +0530), and under `runs` the absolute folder of each held-out set's one run. Those folders are paths on the author's machine, and they stay that way because the held-out guard checks them:

- `eval.run` runs a held-out set only on the code, scoring and truth the file pins, and refuses it in any folder but the one `runs` names. That folder may resume an unfinished run and nothing else.
- `eval.report` gives a run the status frozen, held-out only when the folder its own `eval_run.json` records equals the one `runs` names, and the hashes it ran on are still the ones the file holds.

Do not edit `eval/frozen.json`. Pointing an entry at another folder, or removing one, lets a held-out set run a second time, and a number from that run is a second attempt, not a held-out number, whatever status the report then prints. Changing a hash turns every run on it post hoc. So on another machine the held-out sets cannot be run as held out; what can be checked is that the checkout hashes to the file, as step 1 does.

The author's run folders sit in a temporary folder the system may clear. To keep the runs scorable, copy the whole `reviews` folder somewhere durable, since a second review is scored against its first beside it, and point `data/eval/runs.json`, not `eval/frozen.json`, at the copies. `eval.report` compares the file with the folder each run's own `eval_run.json` records, so a copied run keeps its status.
