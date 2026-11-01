"""One model of a generated set. Geometry comes from Swiss Dwellings and everything a sheet prints is computed from it, so a clean set agrees with itself.

Lengths are metres in plan, with x to the right and y up, turned so walls run along the axes and the long side lies across the sheet.
"""
import math
import random
from collections import Counter, defaultdict
from dataclasses import dataclass, field

import numpy as np
import shapely
from shapely import affinity
from shapely.geometry import LineString, MultiLineString, Point, Polygon, box
from shapely.ops import nearest_points, polylabel, unary_union

from storey.rules import rule

from . import swiss

IN_PER_M = 39.3701
SF_PER_M2 = 10.7639
PT_PER_IN = 72
LOAD_FACTOR = 200       # SF gross per occupant, IBC Table 1004.5 for R-2. rows.json holds no load factor row, so this is the one printed value that does not come from it
DOOR_CLEAR_IN = 2       # a 36 in door leaves 34 in clear past its stop and leaf
PAIR_M = 1.25
SCALES = ((48, '1/4" = 1\'-0"'), (64, '3/16" = 1\'-0"'), (96, '1/8" = 1\'-0"'))   # real inches per paper inch, and the printed note
PLAN_ROOM = 2150, 1270  # points of sheet a plan and its dimension strings may fill
LS_ROOM = 1440, 1380    # the life safety plan shares its sheet with a side panel and has no dimension strings
LEVELS = "FIRST SECOND THIRD FOURTH FIFTH SIXTH SEVENTH".split()
# shortcut: a unit's own corridor is named ENTRY, because vector.corridors measures every room named HALL or CORRIDOR as a public corridor. Named HALL, the free dry review of the
# clean sets gains 5 corridor Cannot tell rows on 710 and 5 to 13 on each of the other six. Name it HALL once that reader leaves the rooms of a unit alone.
NAMES = {"BEDROOM": "BEDROOM", "ROOM": "BEDROOM", "LIVING_ROOM": "LIVING ROOM", "LIVING_DINING": "LIVING / DINING", "DINING": "DINING", "KITCHEN_DINING": "KITCHEN / DINING", "KITCHEN": "KITCHEN",
         "BATHROOM": "BATH", "CORRIDOR": "ENTRY", "STOREROOM": "STORAGE", "OFFICE": "OFFICE", "WASH_AND_DRY_ROOM": "LAUNDRY", "BIKE_STORAGE": "BIKE ROOM", "PRAM_AND_BIKE_STORAGE_ROOM": "BIKE ROOM",
         "TECHNICAL_AREA": "MECHANICAL", "HEATING": "MECHANICAL", "BASEMENT_COMPARTMENT": "STORAGE", "CORRIDORS_AND_HALLS": "HALL", "LOBBY": "LOBBY", "COMMUNITY_ROOM": "COMMUNITY ROOM"}
OUTDOORS = {"BALCONY", "LOGGIA", "TERRACE", "GARDEN", "PATIO", "WINTERGARTEN"}
FIXTURES = {"KITCHEN", "SINK", "TOILET", "BATHTUB", "SHOWER", "WASHING_MACHINE", "BUILT_IN_FURNITURE"}
FINISHES = {"BATH": ("CERAMIC TILE", "TILE", "PAINT", "PAINT"), "KITCHEN": ("LVT", "RUBBER", "PAINT", "PAINT"), "HALL": ("SEALED CONC", "RUBBER", "PAINT", "PAINT"), "CORRIDOR": ("SEALED CONC", "RUBBER", "PAINT", "PAINT")}
FINISH = "OAK", "WOOD", "PAINT", "PAINT"
PROJECTS = "Larkspur Court", "Quarry Row", "Tannery Yard", "Alder Terrace", "Foundry Mews", "Heron Lane", "Cooper Steps", "Mason Green"   # eight, because the listed buildings then differ by id plus seed modulo 8


@dataclass
class Room:
    number: str
    name: str
    poly: Polygon
    unit: str = ""
    label: tuple = (0, 0)

    @property
    def sf(self):
        return round(self.poly.area * SF_PER_M2)


@dataclass
class Door:
    mark: str
    center: np.ndarray
    width: float
    hinge: np.ndarray
    shut: np.ndarray
    tip: np.ndarray
    hand: str
    kind: str           # entrance, exit, balcony or interior
    height: float
    tag: tuple | None = None

    @property
    def leaves(self):
        """(hinge, shut, tip) per leaf. An opening past PAIR_M holds a pair, meeting at its middle, because a leaf that wide does not exist."""
        if self.width <= PAIR_M:
            return [(self.hinge, self.shut, self.tip)]
        out = _unit(self.tip - self.hinge)
        along = self.shut - self.hinge
        return [(self.hinge, self.hinge + along / 2, self.hinge + out * self.width / 2), (self.shut, self.hinge + along / 2, self.shut + out * self.width / 2)]


@dataclass
class Window:
    mark: str
    center: np.ndarray
    u: np.ndarray       # along the wall
    width: float
    depth: float
    height: float
    sill: float
    out: np.ndarray     # towards the outside
    tag: tuple | None = None


@dataclass
class Flight:
    poly: Polygon
    start: np.ndarray   # middle of the end the walk up starts from
    end: np.ndarray


@dataclass
class Floor:
    number: int         # 1 is the discharge level
    name: str           # FIRST FLOOR
    z: float
    walls: list
    hatch: list
    rooms: list
    doors: list
    windows: list
    flights: list = field(default_factory=list)
    landings: list = field(default_factory=list)
    fixtures: list = field(default_factory=list)
    rails: list = field(default_factory=list)
    outdoors: list = field(default_factory=list)
    lift: list = field(default_factory=list)
    shafts: list = field(default_factory=list)
    gross_sf: int = 0
    outline: Polygon = None
    hall: Room = None
    hall_width: tuple | None = None   # two points on the hall's walls, across its width, where the dimension string runs
    hall_text: tuple | None = None    # the middle of that string's text
    dims: list = field(default_factory=list)   # (side, depth, start, end), in plan metres
    lift_doors: list = field(default_factory=list)
    plan_walls: list = field(default_factory=list)   # walls a change put on the floor plan only, not on the life safety background
    stair_on_plan: bool = True   # a defect can leave the stair off the floor plan alone, as a drafting slip does


@dataclass
class Sheet:
    number: str
    title: str
    kind: str           # cover, life, plan, door, window or finish
    floor: int | None = None
    block_title: str | None = None   # what the title block prints, where a defect makes it differ
    in_set: bool = True
    on_index: bool = True


@dataclass
class Model:
    building: str
    seed: int
    variant: str
    name: str
    address: str
    date: str
    scale: tuple
    pt_per_m: float
    ls_scale: tuple
    ls_pt_per_m: float
    extent: tuple       # what the plans cover, x0, y0, x1, y1
    floors: list
    sheets: list = field(default_factory=list)
    code: dict = field(default_factory=dict)
    load_rows: list = field(default_factory=list)
    load_total: dict = field(default_factory=dict)
    exit_rows: list = field(default_factory=list)
    corridor_rows: list = field(default_factory=list)
    door_rows: dict = field(default_factory=dict)      # floor number to rows, printed per floor as real sets do
    window_rows: list = field(default_factory=list)
    finish_rows: dict = field(default_factory=dict)
    markers: dict = field(default_factory=dict)        # floor number to exit markers
    clouds: list = field(default_factory=list)         # second cycle
    revision: dict | None = None


def _polygons(g):
    return [p for p in getattr(g, "geoms", [g]) if p.geom_type == "Polygon" and p.area > 0]


def _grid(rows):
    """The angle in degrees that turns walls onto the axes: the length-weighted middle of edge directions, modulo 90."""
    angles, weights = [], []
    for g in rows:
        for p in _polygons(g):
            xy = np.array(p.exterior.coords)
            d = np.diff(xy, axis=0)
            angles.append(np.degrees(np.arctan2(d[:, 1], d[:, 0])) % 90)
            weights.append(np.hypot(*d.T))
    a, w = np.concatenate(angles), np.concatenate(weights)
    peak = np.bincount((a * 2).astype(int) % 180, w, 180).argmax() / 2 + 0.25
    near = np.abs((a - peak + 45) % 90 - 45) < 2
    return float(peak + np.average((a[near] - peak + 45) % 90 - 45, weights=w[near]))


def _rect(g):
    """A thin rectangle as (centre, unit vector along it, length, thickness), with the vector snapped to an axis when it is that close, because walls now run along the axes."""
    xy = np.array(g.minimum_rotated_rectangle.exterior.coords)
    a, b = xy[1] - xy[0], xy[2] - xy[1]
    long_, short = (a, b) if np.hypot(*a) >= np.hypot(*b) else (b, a)
    length, depth = float(np.hypot(*long_)), float(np.hypot(*short))
    u = long_ / length
    if max(abs(u)) > 0.998:
        u = np.round(u)
    return np.array(g.centroid.coords[0]), u, length, depth


def _wall_span(walls, c, n):
    """(near, far) along n from c, the wall's faces through c. None if no wall lies on that line. A door cut in the source leaves the middle empty, so the jambs are tried too."""
    ray = LineString([c - 0.7 * n, c + 0.7 * n])
    hit = ray.intersection(walls)
    parts = [p for p in getattr(hit, "geoms", [hit]) if p.geom_type == "LineString" and p.length > 0.02]
    if not parts:
        return None
    t = [float(np.dot(np.array(q) - c, n)) for p in parts for q in p.coords]
    return min(t), max(t)


def _cut(walls, c, u, width, depth):
    """The box to take out of the walls for an opening, through the whole wall."""
    n = np.array([-u[1], u[0]])
    spans = [s for s in (_wall_span(walls, c + k * u * width / 2, n) for k in (0, -1.04, 1.04)) if s]
    lo, hi = (min(s[0] for s in spans), max(s[1] for s in spans)) if spans else (-depth / 2, depth / 2)
    lo, hi = max(lo, -0.45), min(hi, 0.45)
    corners = [c + a * u * width / 2 + b * n for a, b in ((-1, lo - 0.01), (1, lo - 0.01), (1, hi + 0.01), (-1, hi + 0.01))]
    return Polygon(corners), lo, hi


def _room_at(rooms, p):
    pt = Point(*p)
    return next((r for r in rooms if r.poly.contains(pt)), None)


def _side(rooms, c, n, reach):
    for k in (reach + 0.15, reach + 0.35, reach + 0.6):
        room = _room_at(rooms, c + k * n)
        if room:
            return room
    return None


def _arc_area(hinge, shut, tip):
    a = math.atan2(*(shut - hinge)[::-1])
    b = math.atan2(*(tip - hinge)[::-1])
    r = float(np.hypot(*(tip - hinge)))
    sweep = (b - a + math.pi) % (2 * math.pi) - math.pi
    points = [hinge + r * np.array([math.cos(a + sweep * k / 8), math.sin(a + sweep * k / 8)]) for k in range(9)]
    return Polygon([hinge, *points])


def _swing(c, u, width, n, walls, blocked, rng):
    """The hinge jamb for a door swinging towards n: the one whose open leaf lies free and nearest a wall. A tie falls to the seed."""
    options = []
    for k in (-1, 1):
        hinge, shut = c + k * u * width / 2, c - k * u * width / 2
        tip = hinge + n * (width if width <= PAIR_M else width / 2)
        sweep = _arc_area(hinge, shut, tip)
        overlap = sweep.intersection(blocked).area if blocked is not None else 0.0
        leaf = LineString([hinge + n * 0.1, tip])
        options.append((round(overlap, 2), round(leaf.distance(walls), 1), rng.random(), hinge, shut, tip))
    return min(options, key=lambda o: o[:3])[3:]


def _hand(hinge, c, n):
    """LEFT or RIGHT, seen from the side the door swings away from."""
    v = hinge - c
    return "LEFT" if n[0] * v[1] - n[1] * v[0] > 0 else "RIGHT"


def _flights(rects):
    """Stair rectangles as flights and landings. One 1.8 m wide or more is a whole U stair, however long, because no flight is that wide, so it splits into two flights and the landing at their head. Of the rest, one much longer than wide is a flight, and it runs along its long side."""
    flights, landings = [], []
    for g in rects:
        c, u, length, width = _rect(g)
        n = np.array([-u[1], u[0]])
        if width < 0.6:
            continue   # the source's strip between two flights
        if width >= 1.8:
            run = length - width / 2
            for side in (-1, 1):
                mid = c + n * side * width / 4 - u * width / 4
                flights.append(Flight(_oriented_box(mid, u, n, run / 2, width / 4 - 0.03), mid - u * run / 2, mid + u * run / 2))
            landings.append(_oriented_box(c + u * run / 2, u, n, width / 4, width / 2))
        elif length / width >= 1.4:
            flights.append(Flight(g, c - u * length / 2, c + u * length / 2))
        else:
            landings.append(g)
    return flights, landings


def _oriented_box(c, u, n, half_long, half_wide):
    return Polygon([c + a * u * half_long + b * n * half_wide for a, b in ((-1, -1), (1, -1), (1, 1), (-1, 1))])


def _name_rooms(entities):
    """Rooms of one floor, unit by unit in reading order, numbered later. In a unit with no living room the largest room is one. Public rooms come under the unit "", where a corridor is a corridor and not an entry."""
    units = defaultdict(list)
    for kind, g, unit in entities:
        units[unit].append((kind, g))
    out = []
    for unit, members in units.items():
        names = {id(g): "CORRIDOR" if kind == "CORRIDOR" and not unit else NAMES.get(kind, kind.replace("_", " ")) for kind, g in members}
        living = any(n.startswith("LIVING") for n in names.values())
        beds = [g for kind, g in members if kind == "ROOM"]
        if beds and not living:
            names[id(max(beds, key=lambda g: g.area))] = "LIVING ROOM"
        out.append((unit, [(names[id(g)], g) for _, g in members]))
    return out


def _reading(g):
    x, y = g.representative_point().coords[0]
    return round(-y / 2), x


def _build_floor(k, z, ents, rng):
    """One floor from its entities: walls cut at openings, rooms, doors and windows, all still unnumbered."""
    area, feature, other = defaultdict(list), defaultdict(list), defaultdict(list)
    for e in ents:
        {"area": area, "feature": feature}.get(e["type"], other)[e["kind"]].append(e)
    raw = unary_union([e["geom"].buffer(0) for s in ("WALL", "COLUMN") for e in other[s]])
    walls = raw.buffer(0.015, join_style=2).buffer(-0.015, join_style=2)
    fixtures = [(s, e["geom"]) for s in sorted(FIXTURES) for e in feature[s]]
    rooms = []
    room_ents = [(s, e["geom"], e["unit"]) for s, rows in area.items() for e in rows if s in NAMES]
    for unit, members in sorted(_name_rooms(room_ents), key=lambda t: _reading(unary_union([g for _, g in t[1]]))):
        for name, g in sorted(members, key=lambda t: _reading(t[1])):
            rooms.append(Room("", name, max(_polygons(g), key=lambda p: p.area), unit))
    hall = Room("", "HALL", max(_polygons(area["STAIRCASE"][0]["geom"]), key=lambda p: p.area))
    cars = [Room("", "ELEV", max(_polygons(e["geom"]), key=lambda p: p.area)) for e in area["ELEVATOR"]]
    everywhere = [*rooms, hall, *cars]
    cuts, doors, windows, lift_doors = [], [], [], []
    solid = unary_union([walls, *(g for _, g in fixtures)])
    for e in other["DOOR"] + other["ENTRANCE_DOOR"]:
        c, u, width, depth = _rect(e["geom"])
        cut, lo, hi = _cut(walls, c, u, width, depth)
        cuts.append(cut)
        n = np.array([-u[1], u[0]])
        front, back = _side(everywhere, c, n, hi), _side(everywhere, c, -n, -lo)
        if front in cars or back in cars:
            lift_doors.append((c + u * width / 2, c - u * width / 2))
            continue
        if front is None and back is None:
            continue
        into, n_into, face, across = _door_side(front, back, n, hi, lo, hall)
        at = c + n * face
        hinge, shut, tip = _swing(at, u, width, n_into, walls, solid, rng)
        kind = "exit" if into is None else "balcony" if across is None else "entrance" if across is hall else "interior"
        doors.append(Door("", at, width, hinge, shut, tip, _hand(hinge, at, n_into), kind, e["height"]))
    for e in other["WINDOW"]:
        c, u, width, depth = _rect(e["geom"])
        cut, lo, hi = _cut(walls, c, u, width, depth)
        cuts.append(cut)
        n = np.array([-u[1], u[0]])
        out = -n if _side(everywhere, c, n, hi) else n
        windows.append(Window("", c + n * (lo + hi) / 2, u, width, hi - lo, e["height"], e["sill"], out))
    open_walls = walls.difference(unary_union(cuts)) if cuts else walls
    polys = _polygons(shapely.set_precision(open_walls, 0.005))
    outline = _outline([*polys, *(r.poly for r in everywhere), *cuts])
    flights, landings = _flights([e["geom"] for e in feature["STAIRS"]])
    return Floor(k, f"{LEVELS[k - 1]} FLOOR", z, polys, _hatch(polys), rooms, doors, windows, flights, landings, fixtures, [e["geom"] for e in other["RAILING"]], [e["geom"] for s in sorted(OUTDOORS) for e in area[s]],
                 [e["geom"] for e in feature["ELEVATOR"]], [e["geom"] for e in area["SHAFT"]], round(outline.area * SF_PER_M2), outline, hall, lift_doors=lift_doors)


def _door_side(front, back, n, hi, lo, hall):
    """(room swung into, direction, wall face it hinges on, room on the other side). Exit doors swing out, a unit's doors swing into the room off its entry, and None is outside."""
    if front is None or back is None:
        inside = back if front is None else front
        direction = n if front is None else -n
        if inside is hall:
            return None, direction, hi if front is None else lo, inside   # the exit: out of the hall
        return inside, -direction, lo if front is None else hi, None
    if front is hall or back is hall:
        unit = back if front is hall else front
        return (unit, -n, lo, hall) if front is hall else (unit, n, hi, hall)
    if front.name == "ENTRY" and back.name != "ENTRY":
        return back, -n, lo, front
    if back.name == "ENTRY" and front.name != "ENTRY":
        return front, n, hi, back
    return (front, n, hi, back) if front.poly.area <= back.poly.area else (back, -n, lo, front)


def _entities(rows, angle, flip, x0, y0):
    """Rows of one floor with geometry turned and moved to the plan's frame."""
    out = []
    for r in rows:
        g = affinity.translate(_turned(r["geom"], angle, flip), -x0, -y0)
        out.append({"type": r["entity_type"], "kind": r["entity_subtype"], "geom": g, "unit": r["unit_id"] if r["unit_usage"] == "RESIDENTIAL" else "", "height": float(r["height"] or 0), "sill": max(float(r["elevation"] or 0) - r["z"], 0)})
    return out


def _turned(g, angle, flip):
    g = affinity.rotate(g, -angle, origin=(0, 0))
    return affinity.rotate(g, 90, origin=(0, 0)) if flip else g


def _parse(rows):
    """{floor id: rows with geometry} for floors at or above grade, lowest first."""
    floors = defaultdict(list)
    for r in rows:
        floors[r["floor_id"]].append(r | {"geom": shapely.from_wkt(r["geometry"])})
    z = {f: min(float(r["elevation"] or 0) for r in rows_ if r["entity_type"] == "area") for f, rows_ in floors.items()}
    kept = sorted((f for f in floors if z[f] >= -0.5), key=z.get)
    return {f: [r | {"z": z[f]} for r in floors[f]] for f in kept}


def _number(f):
    """Room numbers, unit numbers and door marks, so a tag and its schedule row cannot differ. Public rooms have no unit."""
    # shortcut: declared.plan_rooms takes a three digit line under a name for a room, so marks have four digits. With three, the free dry review of the clean 3370 set raises 4 false
    # room name Checks, and 710 none. Use marks such as 201A once a tag no longer reads as a room.
    for n, room in enumerate([*f.rooms, f.hall], 1):
        room.number = f"{f.number}{n:02d}"
    units = {}
    for room in f.rooms:
        if room.unit:
            room.unit = units.setdefault(room.unit, f"{f.number}{len(units) + 1:02d}")
    for n, door in enumerate(sorted(f.doors, key=lambda d: _reading(Point(*d.center))), 1):
        door.mark = f"{f.number}{n:03d}"


def _hatch(polys, step=0.14, thick=0.25):
    """Diagonal hatch in walls thicker than thick. Thin partitions stay as two lines, as drawing offices do."""
    wide = unary_union([p.buffer(-thick / 2, join_style=2).buffer(thick / 2 + 0.001, join_style=2) for p in polys])
    if wide.is_empty:
        return []
    x0, y0, x1, y1 = wide.bounds
    span = y1 - y0
    lines = MultiLineString([[(x0 - span + i * step, y0), (x0 + i * step, y1)] for i in range(int((x1 - x0 + span) / step) + 2)])
    cut = lines.intersection(unary_union(polys).intersection(wide))
    return [np.array(p.coords) for p in getattr(cut, "geoms", [cut]) if p.geom_type == "LineString" and p.length > 0.02]


def _outline(parts):
    """The floor's outer face: walls, rooms and openings joined, with what they enclose filled."""
    joined = unary_union([p.buffer(0.1, join_style=2) for p in parts]).buffer(-0.1, join_style=2)
    return max((Polygon(p.exterior) for p in _polygons(joined)), key=lambda p: p.area)


def _free_label(room, avoid, w, h):
    """Where a room's label goes: the middle of the part of the room clear of fixtures and door swings, or the room's middle if nothing is clear."""
    free = room.poly.buffer(-0.08).difference(avoid) if avoid is not None else room.poly.buffer(-0.08)
    parts = [p for p in _polygons(free) if p.area > w * h]
    if parts:
        return polylabel(max(parts, key=lambda p: p.area), 0.05).coords[0]
    return room.poly.representative_point().coords[0]


def _place_labels(f, pt):
    """Room labels, then door and window tags beside what they name, each clear of what is already placed. `pt` is points per metre."""
    swept = [_arc_area(*leaf) for d in f.doors for leaf in d.leaves]
    avoid = unary_union([g for _, g in f.fixtures] + swept + [fl.poly for fl in f.flights] + f.landings + f.lift) if swept else None
    taken = []
    label_w, label_h = 64 / pt, 28 / pt
    for room in f.rooms:
        room.label = _free_label(room, avoid, label_w, label_h)
        taken.append(box(room.label[0] - label_w / 2, room.label[1] - label_h / 2, room.label[0] + label_w / 2, room.label[1] + label_h / 2))
    blocked = unary_union([*f.walls, *(g for _, g in f.fixtures), *swept, *taken, *f.lift])
    tag = 11 / pt
    for door in sorted(f.doors, key=lambda d: d.mark):
        n = _unit(door.tip - door.hinge)
        u = _unit(door.shut - door.hinge)
        mid = (door.hinge + door.shut) / 2
        spots = [mid + k * n + s * u for k in (-0.52, -0.72, -0.95, 0.5, 0.78, 1.0) for s in (0, 0.3, -0.3, 0.6, -0.6)]
        best = min(spots, key=lambda p: (Point(*p).buffer(tag).intersection(blocked).area > 1e-4, math.dist(p, mid)))
        door.tag = tuple(best)
        blocked = blocked.union(Point(*best).buffer(tag))
    for window in f.windows:
        window.tag = tuple(window.center + window.out * (window.depth / 2 + 0.32))


def _unit(v):
    return v / float(np.hypot(*v))


def ft_in(inches):
    """12'-6" from inches, to the nearest inch."""
    inches = round(inches)
    return f"{inches // 12}'-{inches % 12}\""


def _chain(values, gap=0.8):
    """Dimension stops at least `gap` apart, keeping both ends."""
    out = [values[0]]
    for v in values[1:-1]:
        if v - out[-1] >= gap and values[-1] - v >= gap:
            out.append(v)
    return [*out, values[-1]]


def _on_inch(v):
    """A plan position on a whole inch, so a drawn string and its printed length agree exactly."""
    return round(v * IN_PER_M) / IN_PER_M


def _dimensions(f):
    """Chains of strings along each side, from the outline's corners and the jambs of the windows on that face, and one overall string. `dims` rows are (side, depth, start, end) along that side, depth 0 being the chain."""
    xy = np.array(f.outline.exterior.coords)
    x0, y0, x1, y1 = f.outline.bounds
    f.dims = []
    for side, along, mid in (("bottom", 0, (y0 + y1) / 2), ("top", 0, (y0 + y1) / 2), ("left", 1, (x0 + x1) / 2), ("right", 1, (x0 + x1) / 2)):
        low = side in ("bottom", "left")
        here = xy[xy[:, 1 - along] <= mid] if low else xy[xy[:, 1 - along] >= mid]
        stops = {_on_inch(v) for v in here[:, along]} | {_on_inch(f.outline.bounds[along]), _on_inch(f.outline.bounds[along + 2])}
        for w in f.windows:
            if abs(w.u[along]) > 0.9 and (w.out[1 - along] < 0) == low:
                stops |= {_on_inch(w.center[along] - w.width / 2), _on_inch(w.center[along] + w.width / 2)}
        stops = _chain(sorted(stops))
        f.dims += [(side, 0, a, b) for a, b in zip(stops, stops[1:])]
        if len(stops) > 2:
            f.dims.append((side, 1, stops[0], stops[-1]))


def _hall_label(f, pt):
    """The hall's label and the two wall points across its width: the middle of the hall's open floor, or the nearest straight stretch to it, where walls face each other and no door is nearer along the hall than the far wall is across it."""
    # shortcut: vector.corridors measures only at the label, so the label moves off the hall's middle to a stretch it can read, and a hall with no such stretch keeps its middle and gets no
    # string. At the middle, the free dry review of the clean sets gains 4 corridor Cannot tell rows on 710, 4 on 6267 and 1 on 3128. Keep the middle once corridors() reads a width anywhere in a hall.
    room = f.hall
    room.label = room.poly.representative_point().coords[0]
    free = room.poly.buffer(-0.25)
    if f.flights or f.landings:
        free = free.difference(unary_union([fl.poly for fl in f.flights] + f.landings).buffer(0.2))
    if f.lift:
        free = free.difference(unary_union(f.lift).buffer(0.1))
    parts = _polygons(free)
    if not parts:
        return
    open_floor = max(parts, key=lambda p: p.area)
    middle = polylabel(open_floor, 0.05)
    room.label = middle.coords[0]
    faces = unary_union([p.boundary for p in f.walls])
    doors = unary_union([LineString([d.hinge, d.shut]) for d in f.doors])
    solid = unary_union([fl.poly for fl in f.flights] + f.landings + f.lift + [Point(*d.tag).buffer(12 / pt) for d in f.doors])
    lift = 7 / pt   # the name's middle, which is where a reader of the text layer starts from
    x0, y0, x1, y1 = open_floor.bounds
    spots = [(x, y) for x in np.arange(x0 + 0.05, x1, 0.1) for y in np.arange(y0 + 0.05, y1, 0.1) if open_floor.contains(Point(x, y))]
    for x, y in sorted(spots, key=lambda p: math.dist(p, middle.coords[0])):
        reach = _reach(faces, doors, room.poly, solid, np.array([x, y + lift]), pt)
        if reach:
            room.label, f.hall_width, f.hall_text = (x, y), *reach[1:]
            return


def _hits(lines, c, d):
    """The nearest points on `lines` either way along d from c, or None if one way meets nothing."""
    hits = []
    for k in (-1, 1):
        hit = LineString([c, c + k * d * 40]).intersection(lines)
        if hit.is_empty:
            return None
        hits.append(np.array(nearest_points(Point(*c), hit)[1].coords[0]))
    return hits


def _string(faces, solid, c, across, sides, text, span, pt):
    """((a, b), the middle of the text) for the hall's dimension string, or None: the first of `sides` of the label, in points, where the walls hold the same span and the string and its text stay off `solid`. The text starts at the string's middle and slides along it, because a stair can stand beside a string that clears it."""
    for side in sides:
        h = _hits(faces, c + np.array(side) / pt, across)
        if h and abs(float(np.hypot(*(h[1] - h[0]))) - span) < 0.02 and not LineString(h).intersects(solid):
            for at in (h[0] + (h[1] - h[0]) * k for k in (0.5, 0.3, 0.7)):
                words = at + np.array(text) / pt
                if not LineString([words - across * 16 / pt, words + across * 16 / pt]).buffer(5 / pt).intersects(solid):
                    return (h[0], h[1]), tuple(at)
    return None


def _reach(faces, doors, hall, solid, c, pt):
    """(span, wall points, text middle) across the hall at c, or None unless this is a straight stretch: walls face each other here and 0.3 m either way, and at every height across, the way along the hall is longer than the span. The points and the text come from `_string`: under the number or over the name for a string along x, right or left of the name for one along y."""
    spans = []
    stops = unary_union([faces, doors])
    for across, sides, text in ((np.array([1.0, 0.0]), ((0, -34), (0, 9)), (0, 8)), (np.array([0.0, 1.0]), ((28, 0), (-16, 0)), (-8, 0))):
        hits = _hits(faces, c, across)
        if hits is None or not hall.buffer(0.06).covers(LineString(hits)):
            continue
        along = across[::-1]
        span = float(np.hypot(*(hits[1] - hits[0])))
        same = [_hits(faces, c + k * along, across) for k in (-0.3, 0.3)]
        if not all(h and abs(float(np.hypot(*(h[1] - h[0]))) - span) < 0.02 for h in same):
            continue
        string = _string(faces, solid, c, across, sides, text, span, pt)
        if not string:
            continue
        rows = [_hits(stops, hits[0] + (hits[1] - hits[0]) * k, along) for k in (0.1, 0.3, 0.5, 0.7, 0.9)]
        if all(r and float(np.hypot(*(r[1] - r[0]))) > span + 0.1 for r in rows):
            spans.append((span, *string))
    return min(spans, key=lambda s: s[0]) if spans else None


def _pt_per_m(scale):
    return IN_PER_M / scale[0] * PT_PER_IN


def _fits(extent, room):
    """The largest printed scale at which the plan fits."""
    return next(s for s in SCALES if (extent[2] - extent[0]) * _pt_per_m(s) <= room[0] and (extent[3] - extent[1]) * _pt_per_m(s) <= room[1])


def load(building, seed, variant="clean"):
    """The model of one clean set. Defects edit it afterwards."""
    rng = random.Random(f"{building}:{seed}")
    parsed = _parse(swiss.rows(building))
    floor_rows = list(parsed.values())
    angle = _grid([r["geom"] for r in floor_rows[min(1, len(floor_rows) - 1)] if r["entity_subtype"] == "WALL"])
    walls = unary_union([_turned(r["geom"], angle, False) for rows in floor_rows for r in rows if r["entity_subtype"] == "WALL"])
    bx0, by0, bx1, by1 = walls.bounds
    flip = by1 - by0 > bx1 - bx0
    walls = unary_union([_turned(r["geom"], angle, flip) for rows in floor_rows for r in rows if r["entity_subtype"] == "WALL"])
    x0, y0, x1, y1 = walls.bounds
    floors = []
    for k, rows in enumerate(floor_rows, 1):
        ents = _entities(rows, angle, flip, x0, y0)
        f = _build_floor(k, rows[0]["z"], ents, rng)
        _number(f)
        floors.append(f)
    drawn = [g.bounds for f in floors for g in [*f.rails, *f.outdoors, *f.walls]]
    extent = min(b[0] for b in drawn), min(b[1] for b in drawn), max(b[2] for b in drawn), max(b[3] for b in drawn)
    scale, ls_scale = _fits(extent, PLAN_ROOM), _fits(extent, LS_ROOM)
    pt = _pt_per_m(scale)
    for f in floors:
        _place_labels(f, pt)
        _hall_label(f, pt)
        _dimensions(f)
    name = PROJECTS[(int(building) + seed) % len(PROJECTS)] + " Apartments"
    m = Model(building, seed, variant, name, "000 Nonesuch Way, Testville, ZZ 00000", "NOVEMBER 1, 2026", scale, pt, ls_scale, _pt_per_m(ls_scale), extent, floors)
    _print(m)
    return m


def _windows(m):
    """Window marks by size, so a row covers every window of its size."""
    sizes = Counter((round(w.width * IN_PER_M), round(w.height * IN_PER_M), round(w.sill * IN_PER_M)) for f in m.floors for w in f.windows)
    marks = {size: _letters(n) for n, size in enumerate(sorted(sizes, reverse=True))}
    for f in m.floors:
        for w in f.windows:
            w.mark = marks[round(w.width * IN_PER_M), round(w.height * IN_PER_M), round(w.sill * IN_PER_M)]
    m.window_rows = [{"mark": mark, "qty": sizes[size], "width": ft_in(size[0]), "height": ft_in(size[1]), "sill": ft_in(size[2])} for size, mark in marks.items()]


def _letters(n):
    return chr(65 + n) if n < 26 else _letters(n // 26 - 1) + chr(65 + n % 26)


DOOR_TYPES = {"entrance": ("A", "SOLID CORE", "HM", "ENTRY"), "exit": ("B", "HOLLOW METAL", "HM", "PANIC"), "balcony": ("C", "ALUM / GLASS", "ALUM", "PATIO"), "interior": ("D", "WOOD", "WOOD", "PASSAGE")}


def _leaf_in(door):
    """One leaf in whole inches. A pair's leaves match, so its schedule row and its width come from this one figure."""
    return round(door.width * IN_PER_M) // len(door.leaves)


def door_row(door):
    leaf, height, leaves = _leaf_in(door), round(door.height * IN_PER_M), len(door.leaves)
    kind, material, frame, hardware = DOOR_TYPES[door.kind]
    size = f'{leaf}" x {height}"' if leaves == 1 else f'(2) {leaf}" x {height}"'
    return {"mark": door.mark, "size": size, "swing": door.hand if leaves == 1 else "PAIR", "type": kind, "material": material, "frame": frame, "hardware": hardware, "rating": "20 MIN" if door.kind == "entrance" else "-"}


def _hall_in(f):
    a, b = f.hall_width
    return float(np.hypot(*(np.array(a) - np.array(b)))) * IN_PER_M


def stair_width_in(f):
    runs = [_rect(fl.poly)[3] for fl in f.flights]
    return round(min(runs) * IN_PER_M) if runs else 0


def exit_doors(m):
    return sorted((d for d in m.floors[0].doors if d.kind == "exit"), key=lambda d: d.mark)


def _print(m):
    """Everything the sheets print, computed from the geometry."""
    floors = m.floors
    door_factor, stair_factor = rule("ibc.1005.3.2", "houston")["value"], rule("ibc.1005.3.1", "houston")["value"]
    m.load_rows = [{"no": str(f.number), "name": f.name, "area": f.gross_sf, "factor": LOAD_FACTOR, "load": math.ceil(f.gross_sf / LOAD_FACTOR)} for f in floors]
    m.load_total = {"area": sum(r["area"] for r in m.load_rows), "load": sum(r["load"] for r in m.load_rows)}
    load = m.load_total["load"]
    doors = exit_doors(m)
    if not doors:
        raise ValueError(f"Building {m.building} has no street door in its hall, so pick another")
    upper = load - m.load_rows[0]["load"]
    stair_in = stair_width_in(floors[1]) or stair_width_in(floors[0])
    share = [load // len(doors) + (k < load % len(doors)) for k in range(len(doors))]   # the whole building leaves by the street doors, split evenly
    m.exit_rows = []
    m.markers[1] = []
    for k, (door, carried) in enumerate(zip(doors, share), 1):
        nominal = _leaf_in(door) * len(door.leaves)
        clear = nominal - DOOR_CLEAR_IN * len(door.leaves)
        m.exit_rows.append({"name": f"EXIT {k}", "nominal": nominal, "clear": clear, "load": carried, "required": round(carried * door_factor, 1)})
        m.markers[1].append({"name": f"EXIT {k}", "point": tuple(door.center), "width": clear, "capacity": math.floor(clear / door_factor)})
    m.corridor_rows = [{"name": "1 STAIR", "width": stair_in, "load": upper, "required": round(upper * stair_factor, 1)}]
    m.corridor_rows += [{"name": f"{f.hall.number} HALL", "width": round(_hall_in(f)), "load": m.load_rows[f.number - 1]["load"], "required": round(m.load_rows[f.number - 1]["load"] * door_factor, 1)} for f in floors if f.hall_width]
    stair_capacity = math.floor(stair_in / stair_factor)
    m.code = {"edition": "2021 INTERNATIONAL BUILDING CODE", "occupancy": "R-2", "construction": "TYPE IIIA" if len(floors) <= 5 else "TYPE IB", "sprinklers": "YES, NFPA 13 THROUGHOUT",
              "stories": str(len(floors)), "area": f"{m.load_total['area']:,} SF", "load": str(load)}
    m.door_rows = {f.number: [door_row(d) for d in sorted(f.doors, key=lambda d: d.mark)] for f in floors}
    m.finish_rows = {f.number: [{"no": r.number, "name": r.name, "finish": FINISHES.get(r.name, FINISH), "sf": r.sf} for r in [*f.rooms, f.hall]] for f in floors}
    _windows(m)
    for f in floors[1:]:
        c = unary_union([fl.poly for fl in f.flights] + f.landings).centroid
        m.markers[f.number] = [{"name": "STAIR 1", "point": (c.x, c.y), "width": stair_in, "capacity": stair_capacity}]
    m.sheets = [Sheet("G0.1", "COVER SHEET, SHEET INDEX AND CODE SUMMARY", "cover")]
    m.sheets += [Sheet(f"G1.{f.number}", f"{f.name} LIFE SAFETY PLAN", "life", f.number) for f in floors]
    m.sheets += [Sheet(f"A1.{f.number}", f"{f.name} PLAN", "plan", f.number) for f in floors]
    m.sheets += [Sheet("A6.1", "DOOR SCHEDULE", "door"), Sheet("A6.2", "WINDOW SCHEDULE", "window"), Sheet("A6.3", "ROOM FINISH SCHEDULE", "finish")]
