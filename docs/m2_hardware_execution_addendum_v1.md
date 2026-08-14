# M2 hardware-execution addendum v1

This addendum changes hardware execution only. It does not change the frozen
M2 model, revision, seed, AMP targets, split membership, validation grid,
selection metrics, threshold protocol, optimizer algorithm/hyperparameters,
effective-batch target, or input truncation policy. The frozen
`m2_modernbert_amp_v2.yaml` remains byte-for-byte unchanged.

## Established execution controls

The default maximum length remains 2,048 tokens. When supported, CUDA uses
FP16 autocast with a gradient scaler, MPS uses BF16 autocast without a gradient
scaler, and CPU uses FP32. `--gradient-checkpointing` enables activation
checkpointing with `use_reentrant=false`. A model `use_cache` attribute is set
to false only if that attribute exists; ModernBERT does not currently expose
it, so the action is recorded as not applicable rather than falsely reported
as a configuration change.

The frozen batch fallback is 4, 2, then 1, paired with gradient accumulation
4, 8, then 16. `--initial-train-batch-size 1` starts a fresh process directly at
batch 1 and accumulation 16. This avoids retaining allocations from earlier
same-process fallback attempts while preserving the effective-batch target of
16. A non-semantic heartbeat is printed every 100 training batches with only
epoch, batch/total, and elapsed seconds; it does not log case text, labels, or
per-batch loss and does not affect training.

## Opt-in MPS allocator and shape controls

`--mps-low-watermark-ratio 1.0` sets
`PYTORCH_MPS_LOW_WATERMARK_RATIO=1.0` before the first `torch` import. The
runner accepts only finite positive ratios strictly below the 1.4 unified-memory
default reference. A lower low watermark asks the MPS allocator to attempt
adaptive commits and garbage collection earlier. It does **not** increase the
allocation cap. The runner neither adds nor permits a
`PYTORCH_MPS_HIGH_WATERMARK_RATIO` override; if one is inherited, execution is
rejected. A pre-existing low-watermark value is also rejected unless the same
ratio was explicitly requested on the command line.

`--adamw-foreach-false` explicitly passes `foreach=False` to
`torch.optim.AdamW`. In the installed `xihao_env` PyTorch 2.11.0 build, MPS is
not in the default foreach-supported device list, so the failed MPS runs with
`foreach=None` already resolved to the single-tensor AdamW path. This flag pins
that same path and leaves AdamW's equations, learning rate, weight decay, and
other hyperparameters unchanged; it should not itself be expected to reduce
memory on this installed MPS stack. The requested mode and the optimizer's
observed `foreach` and `fused` parameter-group values are recorded.

`--pad-to-multiple-of 64` retains dynamic padding but rounds each batch's
longest tensor length upward to a 64-token bucket. Original token IDs remain in
order, appended pad positions have attention-mask value zero, and the
truncation cap remains 2,048. Because all permitted caps (2,048, 1,536, and
1,024) are divisible by 64, pre- and post-collation guards prevent padding from
exceeding the recorded cap. This is shape regularization, not guaranteed
memory reduction: it may reduce MPS graph-shape/cache fragmentation, but it can
add 0--63 masked positions and therefore modestly increase forward memory.
Changed padded tensor shapes can also prevent bitwise-identical execution even
though non-padding token content and the scientific selection protocol are
unchanged.

The next clean-process A1 retry restores the frozen 2,048-token limit and
combines all three explicit controls:

```bash
conda run -n xihao_env python src/experiments/09_run_m2_modernbert.py \
  --evaluation A1 \
  --gradient-checkpointing \
  --initial-train-batch-size 1 \
  --mps-low-watermark-ratio 1.0 \
  --adamw-foreach-false \
  --pad-to-multiple-of 64 \
  --local-files-only \
  --force \
  --force-reason "fresh-process 2048 batch-1 retry with low-watermark 1.0, explicit AdamW foreach=false, and 64-token shape buckets after documented allocator-retention OOM"
```

No high-watermark environment setting belongs in this command. `--force`
recoverably archives the latest failed 1,024-token run, including its trial
state, metadata, tokenizer, hashes, and restart rationale, before the new run
writes anything. It refuses to replace completed trial or run artifacts. An
atomic per-evaluation lock prevents concurrent writers; `--break-stale-lock`
must not be added unless an actual orphaned lock has first been confirmed.

## Recorded failure history and length contingencies

Fresh batch-1 attempts with gradient checkpointing have exhausted MPS memory
at all guarded lengths. The 1,536-token failure reported 5.43 GiB of MPS
allocations plus 14.61 GiB of other allocations. The 1,024-token failure
reported 4.43 GiB plus 15.58 GiB, a 20.13 GiB cap, and the same attempted
147.56 MiB allocation. The near-constant total and requested allocation across
lengths motivate testing allocator/cache behavior at the frozen 2,048-token
cap instead of treating a further scientific input reduction as justified.

The existing `--max-length-override {1536,1024}` mechanism remains available
only as an explicitly acknowledged historical contingency. It requires both
`--acknowledge-max-length-reduction` and a non-empty
`--technical-override-rationale`. It is intentionally absent from the next
command.

All execution contexts pin the runner-source hash, exact frozen artifact
hashes, runtime library/hardware fingerprint, requested and observed allocator
environment, precision policy, batch start, checkpointing mode, AdamW
implementation mode, padding bucket, progress policy, and actual maximum
length. These values enter the execution-context hash, so artifacts cannot be
resumed or mixed across modes.
