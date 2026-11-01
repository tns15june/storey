"""A small Responses API agent loop: stateless replay, store false, a turn cap, a spend cap, max_output_tokens on every call.

tool_choice accepts only auto, so a run ends when the model stops calling tools or a cap is hit. Findings come from file_finding tool records in code, never from prose.
"""
import json, sys, time
sys.path.insert(0, "lab")
import metalab as m


def _cost(model, u):
    cached = (u.input_tokens_details.cached_tokens or 0) if u.input_tokens_details else 0
    pi, pc, po = m.PRICES[model]
    return ((u.input_tokens - cached) * pi + cached * pc + u.output_tokens * po) / 1e6, cached


def _keep(item):
    """Output item to replay item. Reasoning stays whole with its encrypted content; everything else only what the input schema needs."""
    d = item.model_dump(exclude_none=True)
    t = d["type"]
    if t == "reasoning":
        return {"type": "reasoning", "encrypted_content": d["encrypted_content"], "summary": d.get("summary") or []}
    if t == "function_call":
        return {"type": "function_call", "call_id": d["call_id"], "name": d["name"], "arguments": d["arguments"]}
    if t == "message":
        return {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": c.get("text", "")} for c in d["content"] if c.get("type") == "output_text"]}
    return None


def output_parts(res, call_id):
    """A tool result as text, or (text, image) as function_call_output content parts. The model cannot see its own call_id, so the result carries it."""
    tag = f"call_id {call_id}\n"
    if isinstance(res, tuple):
        return [{"type": "input_text", "text": tag + res[0]}, {"type": "input_image", "image_url": m.data_url(res[1])}]
    return tag + (res if isinstance(res, str) else json.dumps(res))


def run(model, system, user, tools, handlers, effort="medium", max_turns=12, max_output_tokens=6000, usd_cap=1.0, wrap_up=0):
    """Returns a record with turns, tool log, usd, seconds and stop reason. Handlers return a string, a dict, or (text, image).

    wrap_up: tell the model this many turns before max_turns, or at 80% of usd_cap, to file what it has. A run without it can end with nothing filed.
    """
    items = [{"role": "user", "content": [{"type": "input_text", "text": user}]}]
    rec = dict(model=model, effort=effort, turns=[], tools=[], usd=0.0, stop=None, text=[], t0=time.time())
    warned = False
    for turn in range(max_turns):
        if rec["usd"] >= usd_cap:
            rec["stop"] = "usd_cap"
            break
        if wrap_up and not warned and (turn >= max_turns - wrap_up or rec["usd"] >= .8 * usd_cap):
            warned = True
            rec["wrapped_at"] = turn
            items.append({"role": "user", "content": [{"type": "input_text", "text": "You are nearly out of budget. File every finding you can back now with file_finding, then end."}]})
        m._guard()
        t0 = time.time()
        r = m._retry(lambda: m.client.responses.create(model=model, instructions=system, input=items, tools=tools, tool_choice="auto", store=False,
                                                       include=["reasoning.encrypted_content"], reasoning={"effort": effort}, max_output_tokens=max_output_tokens))
        usd, cached = _cost(model, r.usage)
        rec["usd"] += usd
        m._log(model, usd, input=r.usage.input_tokens, output=r.usage.output_tokens, cached=cached, kind="agent", turn=turn)
        rec["turns"].append(dict(turn=turn, usd=round(usd, 5), input=r.usage.input_tokens, cached=cached, output=r.usage.output_tokens, seconds=round(time.time() - t0, 1), status=r.status))
        calls = []
        for it in r.output:
            k = _keep(it)
            if k:
                items.append(k)
            if it.type == "function_call":
                calls.append(it)
            elif it.type == "message":
                rec["text"].append(" ".join(c.text for c in it.content if c.type == "output_text"))
        if r.status == "incomplete":
            rec["incomplete"] = rec.get("incomplete", 0) + 1
        if not calls:
            rec["stop"] = "no_tool_call"
            break
        for c in calls:
            t1 = time.time()
            try:
                args = json.loads(c.arguments)
                res = handlers[c.name](args)
            except Exception as e:  # a bad argument is the model's to fix, so it goes back as a tool result
                args, res = c.arguments, f"tool error: {type(e).__name__}: {e}"
            rec["tools"].append(dict(turn=turn, call_id=c.call_id, name=c.name, args=args, result=res[0] if isinstance(res, tuple) else res, image=isinstance(res, tuple), seconds=round(time.time() - t1, 2)))
            items.append({"type": "function_call_output", "call_id": c.call_id, "output": output_parts(res, c.call_id)})
    else:
        rec["stop"] = "max_turns"
    rec["seconds"] = round(time.time() - rec.pop("t0"), 1)
    return rec


if __name__ == "__main__":
    # Self-check with one cheap call: a tool returns an image; the model must read a word off it.
    import cv2, numpy as np
    im = np.full((120, 420, 3), 255, np.uint8)
    cv2.putText(im, "HEXAGON", (20, 80), 0, 1.6, (0, 0, 0), 3)
    tools = [{"type": "function", "name": "look", "description": "Returns a picture.", "parameters": {"type": "object", "properties": {}, "required": [], "additionalProperties": False}}]
    rec = run("muse-spark-1.3", "Use the tool, then say the word in the picture.", "What word does the picture show?", tools, {"look": lambda a: ("picture attached", im)}, effort="low", max_turns=3, max_output_tokens=800, usd_cap=0.05)
    print(json.dumps({k: rec[k] for k in ("turns", "stop", "text", "usd")}, indent=1))
    assert any("HEXAGON" in t.upper() for t in rec["text"])
