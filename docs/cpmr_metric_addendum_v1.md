# Contained Partial Match Rate metric addendum v1

Status: **POST-A1, PRE-A2 SECONDARY DIAGNOSTIC ADDENDUM**  
Addendum ID: `sherloc-amp-cpmr-metric-addendum-v1`  
Frozen: 2026-08-14, before M3/M4 A2 TEST inference or A2 result inspection

## Scope and timing

Contained Partial Match Rate (CPMR) is added as a secondary diagnostic for the
17-label SHERLOC Legacy AMP benchmark. CPMR was introduced after observing A1
error behavior as an additional descriptive metric, but before M3/M4 A2
inference and A2 result inspection. It was not part of the original
preregistration.

This addendum changes no model, prompt, demonstration, ontology, A1/A2 split,
prediction, or primary metric. Macro-F1, Micro-F1, per-label Precision/Recall/F1,
Exact-set accuracy, Jaccard, support counts, and their existing bootstrap
confidence intervals remain unchanged.

## Definition

For a target-family silver-reference label set $Y$ and predicted label set
$\hat{Y}$, the case-level CPMR indicator is

```text
CPMR_case = 1 iff |Y_hat| > 0 and Y_hat is a subset of Y;
otherwise CPMR_case = 0.
```

CPMR is deterministic, set-based, insensitive to label order, and unaffected
by duplicate presentation. Predictions and references enter this calculation
through the benchmark's existing canonical ontology mappings.

An empty prediction has CPMR 0. An empty silver-reference set also has CPMR 0:
the only subset of it is the empty prediction, which fails the nonempty
prediction requirement. This gives a safe, explicit result rather than a
zero-denominator calculation.

For a group of $n$ cases, family CPMR is the mean of its binary case indicators:

```text
CPMR = sum(CPMR_case) / n.
```

CPMR is the proportion of cases for which the predicted label set is nonempty
and entirely contained within the silver-reference label set. It captures
conservative partial extraction in which at least one reference-supported
label is recovered without introducing an additional label outside the silver
reference.

CPMR does not require complete recall and therefore is reported only as a
secondary diagnostic alongside F1, Jaccard, and exact-set accuracy. It is not
called accuracy, subset accuracy, exact match, or precision.

## Contained Recall

Contained Recall is defined only when `CPMR_case = 1`:

```text
Contained Recall_case = |Y_hat| / |Y|.
```

The reported Mean Contained Recall is the mean across CPMR-successful cases
only. If a group has no CPMR-successful case, Mean Contained Recall is `N/A`
(`None` in machine-readable metric objects), not zero. This companion value
shows how much of the silver-reference set a conservative contained prediction
recovered.

## Family-specific reporting

CPMR and Mean Contained Recall are calculated independently for Act, Means, and
Purpose. The required case-level fields are:

- `act_cpmr` and `act_contained_recall`;
- `means_cpmr` and `means_contained_recall`;
- `purpose_cpmr` and `purpose_contained_recall`.

The required group-level results are Act CPMR / Act Mean Contained Recall,
Means CPMR / Means Mean Contained Recall, and Purpose CPMR / Purpose Mean
Contained Recall. The diagnostic applies identically to M1--M4 and to A1, each
A2 fold, pooled A2 OOD, and per-jurisdiction A2 results where feasible. No
single global 17-label CPMR replaces these family-specific results.

## Scientific boundary

The reason for adding CPMR is descriptive: it makes conservative partial AMP
extraction visible without relaxing or replacing any preregistered primary
metric. A1 results were already known when this addendum was defined. No M3/M4
A2 TEST result was examined to define it, and no model, prompt, demonstration,
split, ontology, or prediction was changed in response.
