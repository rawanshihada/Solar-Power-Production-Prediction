# Debugging Log

Format: **Symptom → Diagnosis → Fix**

---

## 1. R² stuck at 0.22, and adding trees made it worse

**Symptom.** The first Random Forest scored R² = 0.223. Raising `n_estimators`
from 100 to 200 made it *worse* (0.207). Linear regression scored −0.114 —
worse than predicting a constant. Nothing responded to tuning, which usually
means the problem is not the model.

**Diagnosis.** Two faults stacked on top of each other.

First, the evaluation window. A single chronological 80/20 split puts the
entire test set in late October to December, where mean production is 74 Wh
against 890 Wh in training. R² is measured against the test set's own mean, so
on a window where the target barely moves it collapses. The clue was that the
mean baseline scored −6.03 on that same window — no model can look good against
a reference that broken.

Second, `Month` was a raw integer feature. With that split, months 11 and 12
never appear in training at all, and a tree can only split on values it has
seen. Every November and December row fell into whichever leaf October landed
in, so the model was structurally unable to represent two of the three test
months.

**Fix.** Replaced the single split with `TimeSeriesSplit(n_splits=5)` and made
MAE the selection metric, with RMSE and R² reported alongside. No model changed
— only the measurement — and R² moved from −0.114 to 0.493 for linear
regression and from 0.223 to 0.524 for Random Forest.

---

## 2. Twenty-one consecutive days of zero production in May

**Symptom.** The monthly summary made no physical sense. May averaged 459 Wh
with mean radiation 187 W/m², while June averaged 1,489 Wh with radiation
206 W/m². A 10% difference in sunlight cannot produce a 3× difference in output.

**Diagnosis.** Broke May down day by day. From 3 to 23 May the plant produced
exactly 0 Wh for every hour of every day, including full-sun midday hours. That
is an equipment or metering outage, not weather. Around 500 rows were teaching
the model that high radiation produces nothing — the exact opposite of the
relationship it needed to learn.

The same finding was already visible in the Radiation-vs-Production scatter
plot as a horizontal band of zeros at high radiation. I had missed it because
the points were fully opaque and I had not written a takeaway under the plot.

**Fix.** Wrote a rule that flags any calendar day with zero total production and
mean daily radiation above 30 W/m², so it generalises to future data instead of
hard-coding dates. 504 rows removed: 8,760 → 8,256.

---

## 3. Negative radiation readings

**Symptom.** `describe()` reported a minimum radiation of −9.3 W/m², across
4,464 rows.

**Diagnosis.** Irradiance cannot be negative. This is a sensor offset at night,
not a physical measurement. It had been sitting in the `describe()` output from
the first day — it only became a finding once I asked what the minimum meant.

**Fix.** Clipped radiation at zero during cleaning, and applied the identical
clip inside the API so serving matches training. Predictions are clipped at zero
too, since production cannot be negative either.

---

## 4. Identical code returned two different scores

**Symptom.** The results table gave linear regression MAE = 578.9. Re-running
the same cell later gave 505.6. Linear regression is deterministic — it has no
random seed and must return the same number every time.

**Diagnosis.** The kernel held state from earlier runs. `X` had been built with
`df.drop(columns=[...])`, which takes whatever columns happen to exist in the
DataFrame, so columns added by an earlier experiment were silently included as
features. Random Forest absorbed the difference; linear regression did not.

**Fix.** Restart & Run All, and replaced `drop()` with an explicit `FEATURES`
list naming the inputs and their order. That list is now saved into
`metadata.json` and read by the API, so the feature contract is defined in one
place instead of being inferred. From then on, "a deterministic model returned
two different numbers" is treated as a state problem, not a code problem.

---

## 5. `NameError` on a clean kernel restart

**Symptom.** After a kernel restart the notebook failed at the results cell with
`NameError: name 'cv_score' is not defined`, and separately at the feature cell
with `NameError: name 'np' is not defined`.

**Diagnosis.** `cv_score` and `tscv` had been defined interactively during an
earlier session but never written into a cell. The notebook ran only because
those objects were still resident in kernel memory — the committed file could
not reproduce itself. Anyone cloning the repository would have hit the same
error immediately.

**Fix.** Added an explicit cell defining `tscv` and `cv_score`, placed before
its first use, and restored the deleted cell that trains and saves `v1` so that
`models/v1/` is reproducible rather than an orphaned artefact. `Restart Kernel
and Run All Cells` is now the check before every commit — if the notebook does
not run top to bottom on an empty kernel, it does not work.

---

## 6. A 29 MB model artefact in a 400 KB dataset repository

**Symptom.** The repository weighed 37 MB while the dataset itself is 400 KB.

**Diagnosis.** `joblib.dump()` was called with no compression on an
unconstrained forest. Fully grown trees on 8,000 hourly rows memorise nearly
every row, so the tree structure dominates the file.

**Fix.** Added `compress=3` and kept the depth limit that the tuning search
selected. The promoted artefact dropped to 4.8 MB. Note that removing the old
blob from the working tree is not enough — it stays in git history until the
history is rewritten.

---

## 7. Tuning constrained the model instead of enlarging it

**Symptom.** The randomised search selected `n_estimators=200` (fewer than the
300 in use), `max_depth=16` instead of unlimited, `min_samples_leaf=5` instead
of 1, and `max_features=0.8`. Every chosen value *reduces* model capacity, and
MAE improved from 470.7 to 462.7.

**Diagnosis.** The original configuration was overfitting, not underpowered.
This is consistent with the earlier observation in entry 1: raising trees from
100 to 200 made performance worse. Extra capacity was never the missing piece,
which is why tuning in that direction had never helped.

**Fix.** Kept the constrained configuration. The depth limit acts as
regularisation and also shrinks the artefact, so the smaller model is both more
accurate and cheaper to serve.

---

## Open issue: February

February shows 49% zero-production daylight hours. Unlike May, the zeros are
scattered within days rather than spanning whole days, so the outage rule does
not flag them. It may be a partial outage or snow cover on the panels.
**Unresolved** — recorded here rather than removed, since dropping data on a
guess is worse than keeping it and saying so.
