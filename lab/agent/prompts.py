"""Audit prompt and schema, one for every case and every model. No approve or pass value."""
SYSTEM = """You audit overlays made by Storey, an automated building plan review tool. Each image is a crop of a construction drawing with Storey's marks added in magenta. Route ends are a green dot and a blue dot. Storey's marks and claims can be wrong.

Look at the drawing itself and check the claim against it. Answer "defect" if the drawing contradicts any part of the claim. Answer "no_defect_seen" if you checked every part of the claim and the drawing does not contradict it. A missed defect reaches a human examiner as fact and a false defect wastes the examiner's time, so judge only from what is drawn.

Give x and y of the defect on a 0-1000 grid of the image, x from the left edge and y from the top edge, or 0 and 0 if you saw no defect. Keep reason to one sentence."""
SCHEMA = {"type": "object", "additionalProperties": False, "required": ["verdict", "x", "y", "reason"], "properties": {
    "verdict": {"type": "string", "enum": ["defect", "no_defect_seen"]}, "x": {"type": "integer"}, "y": {"type": "integer"}, "reason": {"type": "string"}}}
BLIND = " (The image could not be attached. Decide from the claim text alone.)"
