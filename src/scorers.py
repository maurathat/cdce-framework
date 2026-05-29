"""
CDCE Scorer Registry — maps task_id to (score, success, graded) grading functions.

Lifted from test_6_3_outcome.py's ad-hoc scoring. Each scorer takes the
model's raw output text and the input_obj dict from the store, and returns
(score: float, success: bool, graded: bool).

score:   1.0 = correct, 0.0 = wrong, partial credit possible per task.
success: True iff the answer is correct within tolerance.
graded:  True if the scorer produced a real measurement. False if the task
         has no objective scorer (score=0.0 is a placeholder, not a measurement).

WARNING: 6 of 9 tasks lack objective scorers and return (0.0, False, graded=False).
Aggregate success rates over the full corpus will be dominated by ungraded
traces — filter to graded=True before computing success%.
"""
import re


def _numbers_from_text(text, expected_len=None):
    """Extract a list of floats from the model's answer.

    When expected_len is provided, walk bracketed lists from the end and
    return the last one whose number-count matches. This avoids picking up
    step-numbering artifacts like [1] that appear after the real answer.
    Falls back to the last bracketed list if no length-match exists."""
    lists = re.findall(r"\[([^\[\]]+)\]", text)
    if not lists:
        return [float(v) for v in re.findall(r"-?\d+\.?\d*", text)]
    if expected_len is not None:
        for span in reversed(lists):
            nums = re.findall(r"-?\d+\.?\d*", span)
            if len(nums) == expected_len:
                return [float(v) for v in nums]
    span = lists[-1]
    return [float(v) for v in re.findall(r"-?\d+\.?\d*", span)]


def _parse_input_list(prompt):
    """Extract the first bracketed list of numbers from a prompt string."""
    m = re.search(r"\[([^\[\]]+)\]", prompt)
    if not m:
        return None
    try:
        return [float(v) for v in re.findall(r"-?\d+\.?\d*", m.group(1))]
    except ValueError:
        return None


def _check_list(answer_text, truth, tol=1e-2):
    """Check a numeric list answer against ground truth.
    Returns (score, success, graded=True)."""
    got = _numbers_from_text(answer_text, expected_len=len(truth))
    if len(got) != len(truth):
        return 0.0, False, True
    if all(abs(a - b) <= tol for a, b in zip(got, truth)):
        return 1.0, True, True
    return 0.0, False, True


# ---- per-task scorers ----

def score_trans_nl_code(input_obj, output):
    """Cumulative average. Parses input list from prompt, computes truth."""
    nums = _parse_input_list(input_obj.get("prompt", ""))
    if nums is None:
        return 0.0, False, False
    s = 0.0
    truth = []
    for i, n in enumerate(nums):
        s += n
        truth.append(round(s / (i + 1), 4))
    return _check_list(output, truth)


def score_pred_code(input_obj, output):
    """Cumulative sum. Parses input list from prompt, computes truth."""
    nums = _parse_input_list(input_obj.get("prompt", ""))
    if nums is None:
        return 0.0, False, False
    s = 0.0
    truth = []
    for n in nums:
        s += n
        truth.append(s)
    return _check_list(output, truth)


def score_pred_math(input_obj, output):
    """Sequence prediction: 2, 6, 14, 30, 62 -> 126, 254, 510.

    INSTANCE-LOCKED: this scorer grades only the canonical §6.4 instance
    (the 2^n - 2 sequence). The rule (a(n) = 2*a(n-1) + 2) is not
    recoverable from the prompt structure alone — it requires recognizing
    the mathematical pattern. If the prompt instance changes, this scorer
    must be updated manually.
    """
    return _check_list(output, [126, 254, 510])


def _extract_tour_distance(output):
    """Extract the model's stated total tour distance from opt_routing output.

    Priority order (most-specific first, plausibility-filtered):
    1a. "total distance/length/cost: N" — explicit multi-word marker.
    1b. "= N miles" anchored to end-of-line.
    1c. "total: N" or "total = N" only if NOT followed by arithmetic
        continuation (+, -, *, /, or another digit). Avoids matching
        "Total: 10+8+7+6+8+16+15 = 70".
    Within each level, take the LAST match. Across levels, return on
    first match. All Priority 1 results are filtered to [40, 200] —
    an implausible marker match falls through to Priority 2.

    2. Last bare number in [40, 200] range.
    3. None if nothing parseable.
    """
    PLAUSIBLE_MIN, PLAUSIBLE_MAX = 40, 200

    # Priority 1a: explicit multi-word markers
    for pattern in [
        r"total\s+distance\s*[:=]\s*(\d{1,3})",
        r"total\s+(?:cost|length|route)\s*[:=]\s*(\d{1,3})",
        r"tour\s+length\s*[:=]\s*(\d{1,3})",
    ]:
        matches = re.findall(pattern, output, re.IGNORECASE | re.MULTILINE)
        plausible = [float(m) for m in matches if PLAUSIBLE_MIN <= float(m) <= PLAUSIBLE_MAX]
        if plausible:
            return plausible[-1]

    # Priority 1b: "= N miles" anchored to end-of-line
    matches = re.findall(
        r"=\s*(\d{1,3})\s*(?:miles|mi)?\s*$", output, re.IGNORECASE | re.MULTILINE,
    )
    plausible = [float(m) for m in matches if PLAUSIBLE_MIN <= float(m) <= PLAUSIBLE_MAX]
    if plausible:
        return plausible[-1]

    # Priority 1c: bare "total: N" only if NOT followed by arithmetic
    matches = re.findall(
        r"total\s*[:=]\s*(\d{1,3})(?!\s*[+\-*/\d])", output, re.IGNORECASE | re.MULTILINE,
    )
    plausible = [float(m) for m in matches if PLAUSIBLE_MIN <= float(m) <= PLAUSIBLE_MAX]
    if plausible:
        return plausible[-1]

    # Priority 2: last bare number in plausible range
    all_nums = re.findall(r"\b(\d{1,3})\b", output)
    plausible = [float(n) for n in all_nums if PLAUSIBLE_MIN <= float(n) <= PLAUSIBLE_MAX]
    if plausible:
        return plausible[-1]

    return None


# Optimal tour length for the canonical §6.4 opt_routing instance.
# Depot -> A -> C -> D -> E -> F -> B -> Depot = 70 miles.
# Verified by brute-force over all 720 permutations of 6 stops.
OPT_ROUTING_OPTIMAL = 70


def score_opt_routing(input_obj, output):
    """Binary grading matching the original §6.4: success = (distance == optimal).

    INSTANCE-LOCKED: grades only the canonical §6.4 routing instance
    (7 nodes, 18 edges, optimal = 70). If the task instance changes,
    OPT_ROUTING_OPTIMAL must be updated.
    """
    dist = _extract_tour_distance(output)
    if dist is None:
        return 0.0, False, False
    success = abs(dist - OPT_ROUTING_OPTIMAL) < 0.5
    return (1.0 if success else 0.0), success, True


# ---- structural scorer (fallback for tasks without ground truth) ----

def score_structural(input_obj, output):
    """No ground truth available. Returns (0.0, False, graded=False).
    score=0.0 is a JSON-safe placeholder, not a measurement.
    Consumers must filter on graded=True before computing statistics."""
    return 0.0, False, False


# ---- registry ----

SCORERS = {
    "trans_nl_code": score_trans_nl_code,
    "pred_code":     score_pred_code,
    "pred_math":     score_pred_math,
    "opt_routing":   score_opt_routing,
}


def score(task_id, input_obj, output):
    """Look up scorer by task_id; return (score, success, graded).

    Signature: score(task_id: str, input_obj: dict, output: str)
               -> (float, bool, bool)
    """
    fn = SCORERS.get(task_id, score_structural)
    return fn(input_obj, output)
