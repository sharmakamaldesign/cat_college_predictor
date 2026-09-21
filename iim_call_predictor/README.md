# iim_call_predictor

Predicts IIM shortlisting calls from a candidate's academics and CAT
percentiles. Currently implements two colleges:

- **IIM Ahmedabad** (PGP 2027-29, CAT-2026) — full score calculation (AR /
  NCS) + CAT cutoff check + call prediction.
- **IIM Mumbai** (MBA 2026-2028, CAT-2025) — eligibility + CAT percentile
  cutoff check + call prediction only (no AR/APWE score — see below).

Every college's output includes a `call` (`true`/`false`) field, but this is
always a **fixed-threshold rule**, never a probability/ML prediction. The
architecture is plug-in based so more IIMs can be added later without
touching shared code.

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
  cli.py
  tests/
    test_iima.py
    test_iimm.py
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
