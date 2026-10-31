box = {"type": "object", "additionalProperties": False, "required": ["label", "x_min", "y_min", "x_max", "y_max"], "properties": {"label": {"type": "string"}, "x_min": {"type": "integer"}, "y_min": {"type": "integer"}, "x_max": {"type": "integer"}, "y_max": {"type": "integer"}}}
arr = {"type": "array", "items": box}
schema = {"type": "object", "additionalProperties": False, "required": ["sheet_floor_label", "stairs", "shafts", "exit_doors", "corridors"], "properties": {"sheet_floor_label": {"type": "string"}, "stairs": arr, "shafts": arr, "exit_doors": arr, "corridors": arr}}
P = ("You are an object grounding expert reading one architectural floor plan. Provide bounding boxes for every instance of: "
     "\"stairs\" (each flight or stair enclosure with treads, interior or exterior), \"shafts\" (elevators, lifts, dumbwaiters, duct or pipe shafts, chimney flues drawn as small enclosures), "
     "\"exit_doors\" (doors in the exterior wall leading outside), \"corridors\" (main long corridors). Do not miss any. "
     "Coordinates are integers on a 0-1000 grid of this image, x from the left edge, y from the top edge. Use the floor label as printed on the sheet for sheet_floor_label, or empty if none is printed.")
