# LLM AMP technical-failure amendment v1

Status: **AUTHORIZED AND FROZEN BEFORE AMENDED REQUESTS**  
Amendment ID: `sherloc-llm-amp-technical-failure-amendment-v1`  
Authorized: 2026-08-14  
Scope: M3/M4 A1 and A2 execution only; this document does not authorize A2 execution.

## Trigger and scientific independence

This amendment was triggered only by technical completion failures observed
during M3 A1 execution:

- otherwise schema-valid multi-label arrays were rejected solely because their
  order differed from the frozen ontology order; and
- Responses API results ended with `status=incomplete` and
  `incomplete_details.reason=max_output_tokens` at the frozen 512-token output
  budget.

No silver-reference test labels were inspected to select or formulate either
change. The amendment is independent of test-label performance. It does not
change the model, prompt, demonstrations, Structured Outputs schema, reasoning
effort, target text, ontology membership, split membership, or scoring
semantics. In particular, the following frozen base artifacts remain unchanged:

- `config/experiments/llm_extraction_amp_v2.yaml`
  (`5da03305ad97b36723c331ade7092147c828365abb32346b14a36726496d330b`)
- `prompts/m3_zero_shot_amp_v2.md`
  (`00b87b84356092b6d01b70f1a495f76c0ebd3ea49eb835a3bd7915a050a23f85`)
- `prompts/m4_six_shot_amp_v2.md`
  (`2d857b1a54b9ed2355558d5f1e8bc7dd3e216e37c5eb7397ffde8d82ee1bfb37`)
- `config/experiments/demo_bank_amp_v1.yaml`
  (`1f6316aa564e44222c5755843544244766daab7344dd002430f365aca235809b`)
- `config/amp_ontology_v1.yaml`
  (`f01a61b5c27f5ed3cc7a8922ddf6ec5aa80f7fea487746d07be358050c5160c1`)
- `data/splits/a1_iid_split_final_v1.csv`
  (`63a739fcb5a1d6af67a1ffc414f5b616a1e2ed7d063f7d34358ac7155803293d`)
- `data/splits/a2_jurisdiction_folds_final_v1.csv`
  (`75ff2d87531bd9b68d2ee6382354d4191229eda4f3b3396d360349ad76e67f67`)

## Rule 1: unordered multi-label canonicalization

For each `acts`, `means`, and `purposes` array, the host must first validate
that the value is an array of strings, every label is unique, and every label
belongs to that field's frozen ontology. Unknown and duplicate values remain
hard failures. After semantic validation, the host canonicalizes the array into
the frozen ontology order before persistence or scoring. Raw structured output
text and the parsed raw response remain preserved separately, so this
presentation-only canonicalization is auditable.

This rule applies uniformly to all future M3/M4 A1 and A2 responses.

## Rule 2: narrowly triggered output-token fallback

Every new case begins with the unchanged base request and
`max_output_tokens=512`. If and only if the returned response has both
`status=incomplete` and
`incomplete_details.reason=max_output_tokens`, the host may retry the same
request with `max_output_tokens=2048`. At most two 2048-token calls are allowed
for a case. A fallback payload must be byte-identical under canonical JSON to
the base payload after removing `max_output_tokens`; all other request fields
must remain unchanged.

HTTP/transient retry handling remains in force, but every actual 2048-token API
call counts toward the two-call fallback ceiling. No other response status,
incomplete reason, schema failure, refusal, or exception can activate this
fallback.

The persisted attempt record must retain the base request identity, the actual
payload hash and output-token budget used for every failed attempt, the exact
incomplete status/reason metadata, and cumulative fallback-call count. A
successful record must retain the same provenance.

Immediately before each permitted 2048-token call, the runner must atomically
reserve that call in a per-case journal. A reservation counts against the
two-call ceiling even if the process ends before it can persist a response; a
resume may use only the unreserved remainder. This deliberately favors a
fail-closed under-spend over a duplicate call after a crash.

## Interrupted M3 A1 recovery

The pre-amendment failure histories for ranks 266 and 1356 already contain
qualifying 512-token responses with the same frozen base request hashes. Their
next calls therefore start directly at 2048; they must not receive another
512-token-only retry. Their previous 512-token histories remain intact. The
two-call 2048 ceiling is cumulative across resumed executions.

Rank 551 has no persisted raw response body that can be recovered. Its recorded
failure is ordering-only, so it receives at most one additional unchanged base
request after Rule 1 is active. If that new response independently meets the
exact Rule 2 trigger, the uniform fallback rule applies.

Rank 551's sole additional 512-token request must likewise be atomically
reserved before the API call. If execution stops after reservation, a resume
must not send a second 512-token recovery request.

All 250 already validated M3 A1 success records are immutable and must be
accepted and skipped byte-for-byte. Only failed or missing cases may be sent.

## Scope boundary

This amendment authorizes completion of missing M3 A1 cases and governs the
already-authorized M4 A1 execution. It establishes the same execution policy
for any later separately authorized M3/M4 A2 run, but it does not itself start
or authorize A2. No API request may be made merely to test this policy; its
regression suite uses offline fake responses only.
