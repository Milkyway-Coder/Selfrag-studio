import os, sys, tempfile
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(HERE, "stubs")); sys.path.insert(0, ROOT); sys.path.insert(0, HERE)
work = tempfile.mkdtemp(); os.chdir(work); os.environ["GROQ_API_KEY"] = "x"
import vision, ingest
from PIL import Image
import io
buf = io.BytesIO(); Image.new("RGB", (300, 200), "white").save(buf, format="PNG"); PNG = buf.getvalue()

def run(script):
    """script: list of results; an Exception instance is raised, a string is returned."""
    calls = []
    def fake(messages, model=None, max_tokens=None, **kw):
        calls.append((model, max_tokens)); r = script.pop(0)
        if isinstance(r, Exception): raise r
        return r
    vision.chat_text, vision.time.sleep = fake, lambda s: calls.append(("sleep", round(s, 1)))
    vision._working = None
    waits = []
    try: out = vision.describe(PNG, "p", 800, on_wait=waits.append)
    except vision.VisionUnavailable as e: out = e
    return out, calls, waits

def test_rate_limit_waits_then_succeeds():
    out, calls, waits = run([Exception("Error code: 429 - Rate limit reached ... Please try again in 7.5s."), "ok"])
    assert out == "ok" and waits == [8.0] and ("sleep", 8.0) in calls, (out, calls)

def test_too_large_retries_smaller_then_next_model():
    big = Exception("Error code: 429 - Request too large for model `x` ... tokens per minute")
    out, calls, _ = run([big, big, "fine"])
    models_tokens = [c for c in calls if c[0] != "sleep"]
    assert out == "fine" and [t for _, t in models_tokens] == [800, 560, 400], models_tokens     # shrinking requests, no sleeping
    out, calls, _ = run([big, big, big, "on-next-model"])
    ms = [c[0] for c in calls if c[0] != "sleep"]
    assert out == "on-next-model" and ms[3] != ms[0], ms

def test_missing_model_moves_on_and_is_remembered():
    out, calls, _ = run([Exception("The model `a` does not exist or you do not have access to it"), "done"])
    assert out == "done" and vision._working == calls[1][0] and calls[0][0] != calls[1][0]

def test_all_fail_message_is_helpful():
    bad = Exception("The model `m` does not exist")
    out, _, _ = run([bad] * 6)
    assert isinstance(out, vision.VisionUnavailable) and "VISION_MODEL" in str(out)

def test_retry_after_parsing():
    assert vision._retry_after("Please try again in 1m12.5s.") == 72.5
    assert vision._retry_after("try again in 450ms") == 0.45
    assert vision._retry_after("no hint") == 10.0

def test_page_selection():
    plain = {"words": 300, "images": 0, "drawings": 2, "tables": 0, "math": 0}
    assert not ingest.needs_vision(plain)
    for k, v in (("images", 1), ("tables", 1), ("drawings", 20), ("math", 3), ("words", 10)):
        assert ingest.needs_vision({**plain, k: v}), k
    assert ingest.math_score("E = mc²  ∑ x ≤ y") >= 2 and ingest.math_score("plain English prose only") == 0

def test_captions_are_clean_text_never_markup():
    junk = ["```latex n\\_cols = 3 \\quad \\text{\\# Number of columns for subplot grid} n\\_rows```",
            "```markdown | Point | Value | Classification | |:--- |:---| x51 | 0.\n```",
            "$labels = [\"Class1\" \\text{ if } x < 0.5 \\text{ else } \"Class2\" \\text{ f"]
    assert all(ingest.clean_caption(j) == "" for j in junk)
    assert ingest.clean_caption("Figure: Hydrological cycle diagram\nShows evaporation.") == "Hydrological cycle diagram"
    assert ingest.clean_caption("```python\nimport os\n```\nThis is a screenshot of Python source code, labeled \"#Program2\". It loads data.") == 'This is a screenshot of Python source code, labeled "#Program2"'
    long = ingest.clean_caption("This figure is a composite of six histograms, each displaying the distribution of a feature across districts.")
    assert len(long) <= 91 and long.endswith("\u2026")
    blk = {"type": "figure", "text": "Figure on page 3", "page": 3, "meta": {"caption": ""}}
    ingest.attach_description(blk, "```markdown\n| a | b |\n```", None)
    assert blk["meta"]["caption"] == "" and "```" not in blk["text"] and blk["text"].startswith("Figure on page 3")

def test_rate_limit_stops_further_vision_calls():
    import make_fixtures
    fx = make_fixtures.make(os.path.join(work, "fx2"))
    calls = []
    def fake(png, prompt, max_tokens, on_wait=None):
        calls.append(prompt[:20])
        raise Exception("Error code: 429 - Rate limit reached for model `m` ... tokens per day (TPD). Please try again in 2h3m.")
    real = vision.describe
    vision.describe = fake
    try:
        r = ingest.parse_file(fx["pdf"], "rl", "paper.pdf", vision_on=True)
    finally:
        vision.describe = real
    assert len(calls) == 1, calls                                    # one failure, then nothing else is attempted
    assert any("rate limit" in w.lower() for w in r["warnings"]), r["warnings"]
    assert r["stats"]["figures"] >= 2 and r["stats"]["vision_pages"] == 0   # the upload still succeeds with plain extraction

if __name__ == "__main__":
    for n, f in list(globals().items()):
        if n.startswith("test_"): f(); print("PASS", n)
