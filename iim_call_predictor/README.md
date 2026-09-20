# iim_call_predictor

Calculates the AWT/PI shortlisting score for IIM Ahmedabad (PGP 2027-29 batch,
CAT-2026) from a candidate's academics, work experience and CAT percentiles.

This project currently implements **only IIM Ahmedabad**. The output includes
a `call` (`true`/`false`) field, but this is a **fixed-threshold rule**
(NCS compared against a per-category cutoff) — not a probability/ML
prediction. The architecture is plug-in based so other IIMs can be added
later without touching shared code.

## Important: results are estimates

IIM Ahmedabad does not publish two numbers this formula needs to normalize
scores:

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

Every `ScoreResult` states which mode was used and which parameter values
were applied, and carries a `warnings` list flagging assumption-based numbers.

## Call prediction

`ScoreResult.call` (`true`/`false`) predicts whether the candidate gets an
AWT/PI call: they must be **eligible**, meet the **CAT-2026 cutoffs**, and
have an **NCS at or above an approximate, per-category cutoff**:

| Category        | NCS call cutoff |
|------------------|-----------------|
| GENERAL / EWS    | 0.92            |
| NC-OBC           | 0.88            |
| SC               | 0.80            |
| ST               | 0.80            |

These thresholds are rough working figures, **not officially published by
IIMA**, and live in [`colleges/iima/reference_params.yaml`](colleges/iima/reference_params.yaml)
(`ncs_call_cutoff`) alongside the other ASSUMPTION-labelled numbers — edit
that file to tune them. `ScoreResult.call_reasons` explains which condition
drove the `true`/`false` outcome.

## Install

```bash
cd cat_college_predictor
pip install -r requirements.txt
```

## Run

Reference mode (default), human-readable output:

```bash
python -m iim_call_predictor.cli --college iima --input candidate.example.json
```

JSON output:

```bash
python -m iim_call_predictor.cli --college iima --input candidate.example.json --json
```

Pool mode, deriving `avg_top50_AR` / per-discipline top-1% averages from your
own candidate-pool CSV (columns: `tenth_pct, twelfth_pct, ug_pct,
work_ex_months, gender, cat_overall_percentile, ug_discipline`):

```bash
python -m iim_call_predictor.cli --college iima --input candidate.json \
    --mode pool --pool-csv my_pool.csv
```

(You can also run `python iim_call_predictor/cli.py ...` directly, from the
repo root, instead of the `-m` form.)

### Candidate input

See [`candidate.example.json`](../candidate.example.json) for the full input
shape. Required fields: `tenth_pct`, `twelfth_pct`, `ug_pct`,
`work_ex_months`, `gender`, `category`, `cat_overall_percentile`,
`cat_varc_percentile`, `cat_dilr_percentile`, `cat_qa_percentile`,
`ug_discipline`. Optional: `pwd` (defaults to `false`).

## Run the tests

```bash
pytest iim_call_predictor/tests/test_iima.py -v
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
  cli.py
  tests/
    test_iima.py
```

`core` never imports from a specific college, and no college-specific number
lives in Python code — everything is in that college's `config.yaml`. This
keeps next year's policy change (new cutoffs, new bands) a config edit, not
a code change.

## Adding a new IIM

1. Create `colleges/<code>/` (e.g. `colleges/iimb/`).
2. Add that college's `config.yaml` (and a `reference_params.yaml` if it also
   needs pool-dependent assumptions).
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
