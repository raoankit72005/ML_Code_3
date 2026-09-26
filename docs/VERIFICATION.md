# Verification record

## Automatic-switching update — 2026-09-26

- Current regression suite: 38 distinct tests passed, one real-CUDA test skipped.
  This is a 38-test discovery run (37 pass / 1 skip) plus the separately run new
  shutdown-failure recovery regression (1 pass).
- Added checks for invalid machine/GPU combinations before any Studio access, plan/status
  without cloud calls, idempotent shutdown, provider/guard shutdown races, shutdown failure
  with retained active state, source mismatch, controller-on-worker refusal, lock release,
  cumulative budget extension/resume, persistent input paths, and private config generation.
- The uploaded er_sample.zip completed all 21 stages through pipeline.py using the offline
  tiny random encoder and real LoRA, FAISS and CPU LightGBM. Restart skipped every stage.
- Original sample counts: 500 train S1, 1,384 train S2, 1,437 train S3; 300 test S1,
  300 test S2, 300 test S3. Test S1 has 100 each from France, India and the US.
- Official validator with --check-ids: PASS. Both TSVs have exactly 300 original test S1 rows.
  run_report.json contains the dataset, candidate-recall, validation, model and coverage reports.
- Syntax compilation and git diff whitespace checks passed.
- Lightning SDK 2026.9.18 start/stop and Machine metadata were inspected locally.
  stop() raises on an already stopped worker; the controller now handles this and confirms
  stopped state before starting the next phase.
- No paid Studios were started. Real account capacity, production pretrained-model accuracy,
  A100/two-GPU execution, provisioning/shutdown and billing remain unverified. These tests
  do not establish full-dataset runtime, leaderboard quality or a cost guarantee.

The launch configuration now defaults to one A100 for training and two for embeddings.
Use docs/LIGHTNING_QUICKSTART.md for the setup generator, sample/full separation and resume.

## Earlier verification record

## Executed locally

- Installed the pinned requirements-lightning.txt environment; `pip check` found no broken requirements.
- Existing + controller suite: 24 tests run, 23 passed, one real-CUDA test skipped.
- Updated controller/index suite: seven tests passed, including the added parallel-index
  equivalence test. Together these cover 24 passing test cases plus the one CUDA skip.
- Real LoRA forward/backward/checkpointing on a tiny locally initialized Transformer.
- Real FAISS search, lexical retrieval, parallel text indexing/blocking/features, full-data
  CPU LightGBM, macro F0.5 threshold selection and test inference on the uploaded er_sample.zip.
- The smoke run completed all 21 stages. It used 500 training S1 queries with 2,821 training
  references and 300 test S1 queries with 600 test references.
- Official supplied validator, unchanged, run with `--check-ids`: PASS.
  Both matching_results.tsv and candidate_pairs.tsv contained exactly 300 sample S1 rows.
  Reference IDs were checked against all 600 sample test S2/S3 records.
- Restart of that completed sample run skipped all 21 stages and exited successfully.
- Parallel index records and TF-IDF arrays were exactly equal to serial indexing in a separate fixture.
- Controller tests use a fake SDK Studio: machine ordering, stopping on failure, resume without
  relaunching completed phases, estimated-budget exhaustion, and remote command exit status.
- Syntax compilation passed. No challenge data or generated model weights are committed.

## Defects found and addressed during checks

- Optional psutil process telemetry could fail in a PID namespace. It now falls back to peak RSS
  rather than aborting training.
- An incomplete candidate TSV was rejected by the final audit. Candidate export now writes to
  a temporary file, flushes/fsyncs, checks its row count and atomically renames it. The rerun
  passed exact original-ID coverage and the official validator.

## Not verified here

- Real Lightning account permissions, S3 authorization, cloud capacity, cross-hardware persistence,
  actual provisioning/shutdown or actual billing. SDK methods/signatures were checked against
  the pinned lightning-sdk 2026.9.18 installation, but no paid machines were started.
- A100/BF16 execution, two-GPU concurrent embedding writes, or a CUDA LightGBM build. The production
  profile deliberately uses CPU LightGBM. The encoder integration smoke uses CPU and a tiny model.
- Fine-tuning quality of the production multilingual pretrained checkpoint. The sample smoke encoder
  is random and tiny: its score and throughput are not a production benchmark.
- Large-partition IVF-PQ performance at full scale; the small sample uses the exact FAISS path.
- The full 24,229,173-record dataset, full-scale RAM/disk peaks, leaderboard score, a 12-hour finish,
  or a ₹4,000 end-to-end bill. Limits can stop the run safely instead of fabricating completion.

Sample outputs are deliberately placed in sample_output/ and refused by the submission packager.
Do not upload them to the leaderboard. Production predictions require a complete original-data run.
