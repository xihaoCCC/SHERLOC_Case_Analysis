# M4 A2 rank-1340 technical exception addendum v1

Status: **AUTHORIZED AND FROZEN BEFORE EXCEPTION REQUESTS AND A2 METRICS**  
Exception ID: `sherloc-m4-a2-fold1-rank1340-rate-limit-exception-v1`  
Authorized: 2026-08-15  
Scope: M4, A2, Fold 1, search rank 1340 only.

## Trigger and authorization

Rank 1340 returned the exact fallback trigger required by the frozen technical
amendment: a 512-token Responses API result with `status=incomplete` and
`incomplete_details.reason=max_output_tokens`. Both subsequently permitted
2048-token calls ended at the transport/API boundary with HTTP 429 and produced
no model response. Their failure history and reservations remain preserved.

Because neither 2048-token call produced a model response, this one-case
exception permits at most two additional calls using the same 2048-token
payload. It is activated only when the runner validates the exact method,
evaluation, fold, rank, base-request hash, 512-token trigger proof, two prior
2048-token reservations, and two matching HTTP 429/no-model-response events.
The additional calls are cumulative and remain fail-closed after two.

## Scientific invariants

This exception was triggered solely by technical rate limiting before a model
response, not by test-label performance. No silver-reference label was
inspected to formulate it. It changes no model, prompt, Structured Outputs
schema, reasoning effort, demonstration, target text, ontology, split,
normalization, or scoring setting. The base request hash and the 2048-token
payload hash must remain unchanged, apart from the already-authorized
512-to-2048 output-budget field change.

The frozen technical amendment remains unmodified at SHA-256
`363c06abb49390a3cf66d646466313d6f50d655e41b801483063d1b180d7cb84`.
This addendum does not apply to any other case and does not authorize auxiliary
features or Evaluation B.
