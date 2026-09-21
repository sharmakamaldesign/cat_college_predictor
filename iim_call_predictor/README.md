# iim_call_predictor

Predicts IIM shortlisting calls from a candidate's academics and CAT
percentiles. Currently implements five colleges:

- **IIM Ahmedabad** (PGP 2027-29, CAT-2026) — full score calculation (AR /
  NCS) + CAT cutoff check + call prediction.
- **IIM Mumbai** (MBA 2026-2028, CAT-2025) — eligibility + CAT percentile
  cutoff check + call prediction only (no AR/APWE score — see below).
- **IIM Calcutta** (MBA 2026-28, CAT-2025) — eligibility + CAT percentile
  cutoff check (incl. non-negative section raw scores) + a composite-score
  call prediction (CAT term + Class 10/12 + gender diversity — see below).
- **IIM Bangalore** (PGP 2026-28, CAT-2025) — eligibility + CAT percentile
  cutoff check (incl. strictly positive section raw scores) + a pre-PI
  composite-score call prediction (CAT + board scores + work-ex + gender
  diversity — see below).
- **IIM Lucknow** (MBA Entrepreneurship & Innovation 2024-26, CAT-2023) —
  eligibility + CAT percentile cutoff check + a Stage 1b composite-score
  call prediction (CAT + Class 12 + graduation + work-ex + gender
  diversity — see below). CAT track only (GMAT not supported).

Every college's output includes a `call` (`true`/`false`) field, but this is
always a **fixed-threshold rule**, never a probability/ML prediction. The
architecture is plug-in based so more IIMs can be added later without
touching shared code.

## Checking every college at once

`iim_call_predictor.predictor.predict_all_colleges(candidate_data: dict) -> dict`
runs every registered college's own eligibility/cutoff/call check for one
candidate and returns a single aggregated response:

```python
from iim_call_predictor.predictor import predict_all_colleges

predict_all_colleges({
    "tenth_pct": 92, "twelfth_pct": 88, "ug_pct": 75, "work_ex_months": 24,
    "gender": "Female", "category": "GENERAL", "cat_overall_percentile": 99.5,
    "cat_varc_percentile": 96, "cat_dilr_percentile": 96, "cat_qa_percentile": 96,
    "ug_discipline": "Engineering & Technology",
    "cat_varc_raw_score": 30, "cat_dilr_raw_score": 25, "cat_qa_raw_score": 28,
})
```

```json
{
  "success": true,
  "message": "Prediction generated successfully.",
  "data": {
    "eligible_colleges": [
      {"college_name": "IIM Ahmedabad", "city": "Ahmedabad", "predicted_call_probability": "High", "recommendation": "Safe"},
      {"college_name": "IIM Bangalore", "city": "Bangalore", "predicted_call_probability": "High", "recommendation": "Safe"},
      {"college_name": "IIM Calcutta", "city": "Kolkata", "predicted_call_probability": "High", "recommendation": "Safe"},
      {"college_name": "IIM Lucknow", "city": "Lucknow", "predicted_call_probability": "High", "recommendation": "Safe"},
      {"college_name": "IIM Mumbai", "city": "Mumbai", "predicted_call_probability": "High", "recommendation": "Safe"}
    ]
  }
}
```

Notes on this aggregated view:
- A college is **left out of `eligible_colleges`** if the candidate doesn't meet its basic eligibility (e.g. UG% too low) — there's no point predicting a call for a college they can't apply to. A college whose model needs fields the payload doesn't supply (e.g. calling IIMM-only fields against IIMA, or omitting IIMC/IIMB's CAT raw scores) is **skipped and noted under a top-level `warnings` list**, rather than erroring out.
- `predicted_call_probability` / `recommendation` (`High`/`Medium`/`Low` and `Safe`/`Moderate`/`Risky`) are a **deterministic bucketing** of each college's own call decision and how comfortably the candidate clears its threshold internally (≥2% margin → High/Safe, positive but <2% → Medium/Moderate, below threshold → Low/Risky; the raw threshold itself isn't exposed in the response, since it's on a different scale per college — IIMA: NCS ~0-1, IIMM: CAT overall percentile, IIMC/IIMB/IIML: composite score ~0-85/~0-100 — and so isn't meaningful to compare across colleges). A call of `true` against a threshold that hasn't been tuned yet also reads `Medium`/`Moderate`, since no real margin can be assessed — never a probability/ML model, just a labeled version of the same fixed-threshold logic.
- On invalid candidate input, the response is `{"success": false, "message": "...", "data": null}` instead.

Via the CLI (drop `--college` to check every college at once):

```bash
python -m iim_call_predictor.cli --input candidate.example.json
```

## IIM Ahmedabad

IIMA does not publish two numbers its formula needs to normalize scores:

- `avg_top50_AR` — the average raw Academic Rating of the top 50 applicants.
- `top1pct_avg_raw_composite` — the average raw composite score of the top 1%
  of applicants, per UG discipline.

By default the tool runs in **`reference` mode**, using placeholder values for
these from [`colleges/iima/reference_params.yaml`](colleges/iima/reference_params.yaml),
clearly labelled `ASSUMPTION - not official`. The resulting **Normalized
Composite Score (NCS) is therefore an estimate**, not an official score.

If you have a real candidate-pool CSV, run in **`pool` mode** instead: the
tool derives both numbers from that data. No synthetic or fabricated
candidate data ships with this project — pool mode only ever uses data you
supply.

`ScoreResult.call` predicts whether the candidate gets an AWT/PI call: they
must be **eligible**, meet the **CAT-2026 cutoffs**, and have an **NCS at or
above an approximate, per-category cutoff**:

| Category      | NCS call cutoff |
|---------------|-----------------|
| GENERAL / EWS | 0.92            |
| NC-OBC        | 0.88            |
| SC            | 0.80            |
| ST            | 0.80            |

These thresholds are rough working figures, **not officially published by
IIMA**, and live in [`colleges/iima/reference_params.yaml`](colleges/iima/reference_params.yaml)
(`ncs_call_cutoff`) — edit that file to tune them.

### Candidate input (IIMA)

See [`candidate.example.json`](../candidate.example.json). Required:
`tenth_pct`, `twelfth_pct`, `ug_pct`, `work_ex_months`, `gender`, `category`,
`cat_overall_percentile`, `cat_varc_percentile`, `cat_dilr_percentile`,
`cat_qa_percentile`, `ug_discipline`. Optional: `pwd` (defaults to `false`).

## IIM Mumbai

IIM Mumbai's own admission policy states that the Academic Performance &
Work Experience (APWE) score and the Personal Interview (PI) score are used
only in **Stage III, for final selection after the PI** — the PI *call*
itself (Stage II) is based on CAT-2025 percentiles only. Since this project
only predicts the call, **no AR/APWE score is computed for IIM Mumbai** —
`raw_ar`, `normalized_ar`, `raw_composite`, `ncs` and `discipline_used` are
all `null` in its `ScoreResult`.

IIMM's own policy publishes only a Stage I **minimum** CAT percentile table
(overall + VARC/DILR/QA per category, with a separate row for PwD candidates
that overrides their reservation-category row) — see
[`colleges/iimm/config.yaml`](colleges/iimm/config.yaml). It explicitly
states the actual Stage II (PI-shortlisting) cutoffs are usually **higher**
and vary by year depending on how many candidates are called, which it does
not publish in advance.

The `call` prediction itself uses an **externally estimated overall
CAT-2025 percentile range per category** — not published by IIM Mumbai —
from [`colleges/iimm/reference_params.yaml`](colleges/iimm/reference_params.yaml)
(`call_overall_percentile_range`). The midpoint of each range is the
go/no-go threshold:

| Category | Estimated overall percentile range | Midpoint used |
|----------|-------------------------------------|---------------|
| GENERAL  | 97-98                               | 97.5          |
| EWS      | 90-91                               | 90.5          |
| NC-OBC   | 92-93                               | 92.5          |
| SC       | 80-81                               | 80.5          |
| ST       | 68-69                               | 68.5          |

PwD candidates have no updated estimate yet, so they fall back to a
sectional check against the Stage I PWD minimums (`call_cutoff.PWD` in the
same file). Edit `reference_params.yaml` to tune any of these once better
figures are available; only `reference` mode is supported (no `pool` mode)
since there's no meaningful way to derive the undisclosed Stage II bar from
a plain candidate CSV.

### Candidate input (IIMM)

See [`candidate.iimm.example.json`](../candidate.iimm.example.json).
Required: `ug_pct`, `category`, `cat_overall_percentile`,
`cat_varc_percentile`, `cat_dilr_percentile`, `cat_qa_percentile`. Optional:
`pwd` (defaults to `false`) — when `true`, the PWD cutoff row is used
regardless of `category`. `category: "GENERAL"` represents IIMM's own
"OPEN" category label.

## IIM Calcutta

Unlike IIMM, IIM Calcutta's own policy states its Stage II PI/WAT call
**is** based on a published composite formula (Table 2): a CAT-score term
(56%) + Class 10 marks (10 pts, banded) + Class 12 marks (15 pts, banded) +
a gender-diversity bonus (+4 for female/transgender candidates) — so `call`
here uses that composite, not just CAT percentiles. IIMC's Table 4 (final
selection: PI, WAT, academic diversity, work experience) and Table 5
(academic diversity categories) apply only to Stage III, after the PI/WAT —
out of scope, and not implemented.

**Stage I** adds one thing beyond the usual percentile table: the policy
requires **non-negative raw scores in all three CAT sections** (not just
minimum percentiles) to be considered further — enforced via three new
required fields, `cat_varc_raw_score`, `cat_dilr_raw_score`,
`cat_qa_raw_score`.

**Stage II's CAT term** is officially `(candidate's CAT scaled score / max
possible scaled score) * 56`. IIM Calcutta doesn't publish either number,
and this tool doesn't collect a raw scaled score, so the CAT term is
approximated as `(cat_overall_percentile / 100) * 56` — flagged as a rough
proxy (a percentile isn't the same statistic as a scaled-score ratio), not
the official calculation.

**The Stage II composite cutoff** is explicitly "decided... at the sole
discretion of the Institute" and not published. `call` uses the midpoint of
an **externally estimated composite-score range per category** (2026-28,
not published by IIM Calcutta) from
[`colleges/iimc/reference_params.yaml`](colleges/iimc/reference_params.yaml)
(`composite_call_range`):

| Category | Estimated composite range | Midpoint used |
|----------|----------------------------|---------------|
| GENERAL  | 50-56                      | 53.0          |
| EWS      | 47-50                      | 48.5          |
| NC-OBC   | 43-49                      | 46.0          |
| SC       | 38-43                      | 40.5          |
| ST       | 33-36                      | 34.5          |

PwD candidates have no updated estimate yet, so they fall back to a cutoff
of 0 (no additional bar beyond Stage I) — `composite_call_cutoff_pwd_fallback`
in the same file. Edit `reference_params.yaml` to tune any of these once
better figures are available; only `reference` mode is supported (no `pool`
mode).

### Candidate input (IIMC)

Required: `ug_pct`, `category`, `cat_overall_percentile`,
`cat_varc_percentile`, `cat_dilr_percentile`, `cat_qa_percentile`,
`cat_varc_raw_score`, `cat_dilr_raw_score`, `cat_qa_raw_score`, `tenth_pct`,
`twelfth_pct`, `gender`. Optional: `pwd` (defaults to `false`) — when
`true`, the PWD cutoff row is used regardless of `category`. `category:
"GENERAL"` represents IIMC's own "OPEN" category label.

## IIM Bangalore

Like IIMC, IIM Bangalore's own Admission Procedure states the Stage II
PI/WAT call **is** determined by a published composite — the "pre-PI
score" — out of 100: CAT (55, split VARC 19/DILR 21/QA 15) + 10th board
(10) + 12th board (10) + Bachelor's degree (10) + work experience (10,
capped at full credit for ≥36 months) + gender-diversity bonus (5). The
post-PI composite (PI=40, WAT=10, CAT=25, re-weighted board/work-ex) that
determines **final selection**, after the PI/WAT, is out of scope — as are
the COVID-era missing-10th/12th-score re-weighting rule and the
"automatic top-10 qualifier" rule (by total CAT score / adjusted bachelor's
score / professional-course score), both of which need whole-applicant-pool
ranking data this single-candidate tool doesn't have.

**Stage I** requires a **strictly positive** (`> 0`, not merely
non-negative like IIMC) raw score in every CAT section, alongside the
usual percentile table — IIM Bangalore's own numbers, distinct from
IIMM/IIMC's (including its own PwD row: 50/50/50/60).

**The pre-PI composite's official formula** standardizes every component
via a population z-score — `max(0, min(wt, wt/2 + ((val-mean)/sd)*wt/6))` —
against a mean/SD IIM Bangalore doesn't publish. Since this tool doesn't
collect a candidate pool by default, each z-scored component (CAT
sections, 10th, 12th, Bachelor's) is approximated as `(value / 100) *
weight` instead — a rough linear proxy, not the official calculation. Work
experience is **not** approximated — it uses the exact published formula
(`weight * months/36` for `0 < months < 36`, full weight for `months >=
36`).

**The pre-PI composite cutoff** used to call candidates for Stage II is
not published ("a certain number of candidates from each category are
selected"), so `call` uses an **externally estimated, user-supplied
per-category cutoff** from
[`colleges/iimb/reference_params.yaml`](colleges/iimb/reference_params.yaml)
(`pre_pi_call_cutoff`): GENERAL 52.5, EWS 42.0, NC-OBC 43.5, SC 37.5,
ST 28.5, PWD 30.0 — not published by IIM Bangalore. Edit that file to tune
these as better figures become available; only `reference` mode is
supported (no `pool` mode).

This document (2026-28 cycle) is explicitly used as-is for the 2027 (CAT-2026)
cycle too, per your instruction — IIM Bangalore's own disclaimer notes it
may change its process for future cycles without notice.

### Candidate input (IIMB)

Required: `ug_pct`, `category`, `cat_overall_percentile`,
`cat_varc_percentile`, `cat_dilr_percentile`, `cat_qa_percentile`,
`cat_varc_raw_score`, `cat_dilr_raw_score`, `cat_qa_raw_score`, `tenth_pct`,
`twelfth_pct`, `work_ex_months`, `gender`. Optional: `pwd` (defaults to
`false`) — when `true`, the PWD cutoff row is used regardless of
`category`. Candidates whose only qualification is a completed
CA/CS/ICWA/FIAI professional degree (no bachelor's) should supply their
professional-course aggregate percentage as `ug_pct`.

## IIM Lucknow

IIM Lucknow's MBA (Entrepreneurship & Innovation) programme also accepts a
valid **GMAT** score as an alternative to CAT — **only the CAT track is
implemented**; this tool has no GMAT input fields. Like IIMC/IIMB, the
Stage 1b composite **is** what determines the PI/Business-Plan call: CAT
(45) + Class 12 (20) + Graduation (25) + work experience (5, exact
published formula: `min{(months-6)*0.5, 5}` for `months > 6`) +
gender-diversity bonus (5, **Female only** — this policy's own wording is
narrower than the other colleges here, which also credit "Other"). Stage 2
(Business Plan 40% + Interview 60%, 18/60 interview pass mark) determines
**final selection**, after the PI, and is out of scope.

**Two Stage 1b components rely on unpublished applicant-pool statistics**
and are approximated:
- CAT term: officially `(candidate's score / highest CAT score in the
  pool) * 45`, approximated as `(cat_overall_percentile / 100) * 45`.
- Class 12 (12M): officially based on the candidate's Class 12 *percentile
  within the applicant pool* (by board and discipline) — not their raw
  percentage — via `[{max(P,80)-80}/20] * 20`. Approximated by treating
  the raw `twelfth_pct` as if it were already that pool percentile:
  `clamp(twelfth_pct - 80, 0, 20)`.
- Graduation (GM): officially a population z-score by academic discipline;
  approximated as `(ug_pct / 100) * 25`, ignoring the discipline split.

**The Stage 1b composite cutoff** is not published, so `call` uses an
**externally estimated, user-supplied per-category cutoff** from
[`colleges/iiml/reference_params.yaml`](colleges/iiml/reference_params.yaml)
(`pre_pi_call_cutoff`): GENERAL 52, EWS 39, NC-OBC 37.5, SC 30, ST 21.5,
PWD 12 (the source explicitly flags the PWD figure as low-confidence) —
not published by IIM Lucknow. Edit that file to tune these; only
`reference` mode is supported (no `pool` mode).

### Candidate input (IIML)

Required: `ug_pct`, `category`, `cat_overall_percentile`,
`cat_varc_percentile`, `cat_dilr_percentile`, `cat_qa_percentile`,
`twelfth_pct`, `work_ex_months`, `gender`. Note: `tenth_pct` is **not**
needed — unlike every other college here, IIML's composite has no 10th-board
component. Optional: `pwd` (defaults to `false`) — when `true`, the PWD
cutoff row is used regardless of `category`.

## Install

```bash
cd cat_college_predictor
pip install -r requirements.txt
```

## Run

Reference mode (default), human-readable output:

```bash
python -m iim_call_predictor.cli --college iima --input candidate.example.json
python -m iim_call_predictor.cli --college iimm --input candidate.iimm.example.json
python -m iim_call_predictor.cli --college iimc --input candidate.iimc.example.json
python -m iim_call_predictor.cli --college iimb --input candidate.iimb.example.json
python -m iim_call_predictor.cli --college iiml --input candidate.iiml.example.json
```

JSON output:

```bash
python -m iim_call_predictor.cli --college iima --input candidate.example.json --json
```

Pool mode (IIMA only), deriving `avg_top50_AR` / per-discipline top-1%
averages from your own candidate-pool CSV (columns: `tenth_pct,
twelfth_pct, ug_pct, work_ex_months, gender, cat_overall_percentile,
ug_discipline`):

```bash
python -m iim_call_predictor.cli --college iima --input candidate.json \
    --mode pool --pool-csv my_pool.csv
```

(You can also run `python iim_call_predictor/cli.py ...` directly, from the
repo root, instead of the `-m` form.)

## Run the tests

```bash
pytest iim_call_predictor/tests/ -v
```

## Architecture

```
iim_call_predictor/
  core/
    base.py       # abstract CollegeModel interface
    registry.py    # @register_college("code") + get_college(code)
    models.py      # shared CandidateInput / ScoreResult pydantic models
    utils.py       # banding, clamping, top-N average helpers
  colleges/
    iima/
      config.yaml            # all IIMA tables/weights/cutoffs — no magic numbers in code
      reference_params.yaml  # ASSUMPTION pool-dependent parameters
      model.py                # IIMAModel(CollegeModel)
    iimm/
      config.yaml            # official IIMM eligibility + Stage I cutoff table
      reference_params.yaml  # ASSUMPTION Stage II call-cutoff parameters
      model.py                # IIMMModel(CollegeModel)
    iimc/
      config.yaml            # official IIMC eligibility + Stage I/II tables (Tables 1-3)
      reference_params.yaml  # ASSUMPTION Stage II composite call-cutoff parameters
      model.py                # IIMCModel(CollegeModel)
    iimb/
      config.yaml            # official IIMB eligibility + Stage I table + pre-PI weights
      reference_params.yaml  # ASSUMPTION pre-PI composite call-cutoff parameters
      model.py                # IIMBModel(CollegeModel)
    iiml/
      config.yaml            # official IIML eligibility + Stage 1a table + Stage 1b weights
      reference_params.yaml  # ASSUMPTION Stage 1b composite call-cutoff parameters
      model.py                # IIMLModel(CollegeModel)
  predictor.py    # predict_all_colleges(candidate_data) — checks every registered college at once
  lambda_handler.py  # AWS Lambda entry point (auth + delegates to predictor)
  cli.py
  tests/
    test_iima.py
    test_iimm.py
    test_iimc.py
    test_iimb.py
    test_iiml.py
    test_predictor.py
    test_lambda_handler.py
```

`core` never imports from a specific college, and no college-specific number
lives in Python code — everything is in that college's `config.yaml`. Not
every college uses every `CandidateInput` field (e.g. IIMM never reads
`tenth_pct`/`twelfth_pct`/`work_ex_months`/`gender`/`ug_discipline`), so
those fields are optional at the shared-schema level; each college's model
raises a clear error if a field *it* actually needs is missing.

## Adding a new IIM

1. Create `colleges/<code>/` (e.g. `colleges/iimb/`).
2. Add that college's `config.yaml` (and a `reference_params.yaml` if it also
   needs unpublished/assumption-based parameters).
3. Write `colleges/<code>/model.py` with a class subclassing
   `core.base.CollegeModel`, decorated with `@register_college("<code>")`,
   implementing:
   - `check_basic_eligibility(candidate)`
   - `check_cutoffs(candidate)`
   - `compute_score(candidate, params, mode)`
   - `describe_required_inputs()`
4. Import the new subpackage from `colleges/__init__.py` so the decorator
   runs and the college self-registers.
5. Nothing in `core/` needs to change.

A college that needs fields beyond the shared `CandidateInput` schema can
define its own small pydantic model for those extra, college-specific
inputs, without affecting other colleges.

# All college test script
python3 -m iim_call_predictor.cli --input candidate.example.json
python3 -m iim_call_predictor.cli --input candidate.iimc.example.json