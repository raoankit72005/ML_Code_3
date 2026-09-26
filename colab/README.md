# Google Colab single-GPU workflow

Open [ML_Code_1_Colab.ipynb](https://colab.research.google.com/github/raoankit72005/ML_Code_1/blob/main/colab/ML_Code_1_Colab.ipynb), select a GPU runtime and run cells in order. Private repository users can download the notebook and code ZIP while signed in, upload the notebook to Colab and select the code-ZIP method in cell 1.

The notebook installs dependencies in a dedicated interpreter, copies your original ZIP from Drive to local disk, detects GPU/RAM/storage, generates reproducible settings, and launches the full pipeline. Default: LoRA encoder on GPU, CUDA LightGBM. CPU matching or a frozen encoder are explicit alternatives. CPU cleaning/retrieval/features remain CPU operations. No speed or leaderboard improvement is guaranteed; compare observed throughput with your deadline before moving a working SageMaker run.

Colab may disconnect/delete the VM and does not guarantee a particular GPU or session length. Buying Drive storage does not expand local VM storage. At least 20 GiB free disk and 6 GiB available RAM are required just to start; real full-data peak requirements are substantially larger and checked again by stages. This is not an assertion that the whole dataset fits these minimums. All candidate training pairs are retained; LightGBM cannot make resident binned data arbitrarily small through batch loading.

Checkpoints and small logs are copied to a unique Drive run directory about every 3 minutes. Final model files, TF-IDF state, reports, TensorBoard logs, candidate_pairs.tsv and matching_results.tsv are copied at exit. Active SQLite/vector files stay on local disk. Interrupted tensorboard/model outputs should not be treated as complete: verify the final audit. Backup failure prints an error; a final backup failure makes the launcher fail visibly while keeping local results.

**Drive backups are portable artifacts, not full-pipeline resume snapshots.** They exclude potentially huge SQLite, cleaned data and vector caches. The same local runtime can resume compatible work. A deleted VM or a SageMaker-to-Colab move requires an explicit migration of all work state with compatible signatures; do not edit signatures or claim checkpoint-only automatic resume. Use a new backup name after losing a runtime so existing artifacts are preserved.

The runner's process lock prevents two instances of this launcher sharing one work directory. Never launch hybrid.py directly against that directory while the launcher is active. Changes are confined to colab/ and tests; existing SageMaker source/configuration signatures are unaffected by this addition.

Development validation: Python/shell/notebook syntax and standard-library tests of generated settings, bounded atomic backup, and excluded work caches. A Colab GPU and full dataset are not available in the development environment; actual installation/CUDA execution and full-scale runtime remain to be verified on Colab.
