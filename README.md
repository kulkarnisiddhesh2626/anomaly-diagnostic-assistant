# Transaction Anomaly Detection & Diagnostic Assistant

A proof-of-concept tool for the risk & operations team of a **card-issuing bank**. It
watches issuer-side transaction health metrics over time, flags anomalies with an
**explainable statistical detector**, and uses an **LLM to turn detector output into a
plain-language diagnosis** plus a grounded conversational Q&A interface.

The end-to-end flow is exactly the one in the brief:

```
synthetic data  →  detection layer  →  context builder  →  LLM diagnosis  →  chat Q&A
```

The two layers are kept strictly separate: **the detector finds anomalies; the LLM only
explains them.** The LLM never sees raw transactions — it only ever receives a compact,
pre-computed "fact sheet" of numbers the detector produced. That is the core of the
hallucination-control design (see the write-up).

---

## Quick start

```bash
# 1. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. (Optional) generate the synthetic dataset.
#    The app does this automatically on first run, but you can pre-build it:
python -m src.data_generator

# 4. Run the app
streamlit run app.py
```

Then open the URL Streamlit prints (default <http://localhost:8501>).

### Optional: enable the live LLM

The app **runs fully without any API key.** With no key it uses a deterministic,
template-based diagnosis generator so you can demo the entire flow offline. To switch on
real LLM-generated narratives and free-form chat:

```bash
cp .env.example .env
# edit .env and set ANTHROPIC_API_KEY=sk-ant-...
```

The sidebar shows an **LLM: online / offline** badge so you always know which path is
active. Provider/model are configurable via environment variables (see `.env.example`);
swapping to OpenAI/Gemini/local means editing only `src/llm_client.py`.

---

## What the app shows

The UI has three tabs:

1. **Overview** — headline KPIs (approval rate, decline rate, fraud rate, volume), a
   time-series chart with detected incidents highlighted as red bands, and a table of
   flagged events. Sidebar sliders let you regenerate data (days / random seed) and tune
   the detector (z-threshold, minimum hourly volume) live.
2. **Incidents & Diagnosis** — pick an incident to see the exact **fact sheet** passed to
   the LLM (transparent, expandable JSON), the generated **diagnostic narrative**
   (what happened / where it's concentrated / probable root cause / recommended next
   steps), and a bar chart of the decline-reason mix shift.
3. **Ask the data** — a chat box where an analyst can ask follow-ups
   ("Why did approval rates drop on the 10th?", "Which decline codes drove the spike?").
   Answers are grounded in the same pre-computed aggregates.

---

## Repository layout

```
anomaly-diagnostic-assistant/
├── app.py                     # Streamlit UI (the three tabs above)
├── requirements.txt
├── .env.example               # copy to .env to enable the live LLM
├── README.md
├── WRITEUP.md                 # 1–2 page design write-up + architecture diagram
├── data/                      # generated CSV + ground-truth JSON (git-ignored)
└── src/
    ├── config.py              # dimensions, decline codes, paths, detector + LLM settings
    ├── data_generator.py      # synthetic issuer dataset + 5 injected anomalies
    ├── detection.py           # statistical detector + Isolation Forest cross-check
    ├── context_builder.py     # turns an incident into a compact fact sheet for the LLM
    ├── llm_client.py          # thin Anthropic Messages API wrapper (online/offline)
    └── diagnosis.py           # prompts + LLM calls + deterministic offline fallback
```

---

## The data (synthetic)

`src/data_generator.py` produces **45 days of hourly aggregates** (~275k rows, ~2.2M
simulated transactions). The schema follows the brief:

| column | meaning |
|---|---|
| `timestamp` | hour bucket |
| `mcc` | merchant category (5411 Grocery, 5812 Restaurants, 5999 Misc Retail, 4829 Money Transfer, 7995 Gambling) |
| `country` | US (issuer / domestic), GB, IN, DE, BR |
| `channel` | ecom (CNP), pos (CP), contactless (CP) |
| `card_present_flag` | derived from channel |
| `auth_type` | 3DS / non-3DS |
| `decline_reason_code` | one outcome bucket per row — `00_APPROVED` or an ISO-8583-style decline code |
| `txn_count`, `approved_count`, `declined_count`, `fraud_count`, `txn_amount` | the measures |

**One row = one outcome bucket within a (slice, hour).** Approvals carry the synthetic
`00_APPROVED` code; declines carry a real reason code. This keeps a single tidy table while
making "which decline reason drove the spike?" a one-line group-by.

The generator builds realistic **diurnal + weekly seasonality** and per-slice baselines,
then injects **5 anomalies** and records the ground truth in
`data/injected_anomalies.json`:

1. **Issuer/processor outage** (6h, all slices) — `91_ISSUER_UNAVAILABLE` technical
   declines surge, approval rate collapses.
2. **Card-testing fraud attack** (2 days, Misc-Retail / BR / ecom) — `fraud_count` and
   `59_SUSPECTED_FRAUD` spike.
3. **3DS authentication failure** (24h, GB / ecom / 3DS) — `N7_3DS_FAILURE` declines spike.
4. **Cross-border approval drop** (3 days, IN / ecom) — elevated `05_DO_NOT_HONOR` declines.
5. **Gambling volume surge** (8h, MCC 7995) — ~3× transaction volume.

The current detector finds **all five** (100% recall on the injected set).

---

## Detection method (short version)

Full justification is in `WRITEUP.md`. In brief:

- **Rate metrics** (approval / decline / fraud rate) are tested with a **two-proportion
  z-test** against a **pooled seasonal baseline** (same weekday × hour). This is volume
  aware: a 0/3 fraud blip in a tiny bucket scores low, while a real shift over thousands of
  transactions scores high — directly targeting the *low-false-positive* requirement.
- **Volume** is scored with a **Poisson-style** deviation against its seasonal mean.
- **Effect-size floors** (e.g. a rate must move at least ~3pp; a fraud event needs a
  minimum absolute count) suppress statistically-significant-but-trivial noise.
- An **Isolation Forest** runs as a multivariate **cross-check only** — it can corroborate
  but never solely trigger an alert, because it isn't explainable on its own.
- Contiguous flagged hours are merged into **events**, and overlapping events are
  consolidated into **incidents** (one operational story per incident).

Everything the detector emits is a concrete number with a before/during/after context,
which is what makes the downstream LLM explanation trustworthy.

---

## Regenerating / resetting data

```bash
python -m src.data_generator        # rewrites data/transactions.csv + ground truth
```

The `data/` CSV and ground-truth JSON are **git-ignored** — anyone who clones the repo
regenerates them with the command above (deterministic per seed).

---

## Notes / limitations

- This is a **POC**: the focus is a clean, working end-to-end flow, not breadth.
- The offline diagnosis path restates verified detector facts; the online path produces
  richer prose but is held to the same grounded fact sheet.
- See **"What I'd improve with more time"** in `WRITEUP.md`.
- https://anomaly-diagnostic-assistant-inbxjtceuumuzt5wyvbrqc.streamlit.app/
