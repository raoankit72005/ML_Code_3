# Verification scope

The suite includes CPU execution of real LightGBM, a tiny local Transformer with LoRA and cached contrastive backpropagation, embedding persistence, FAISS retrieval, hybrid pair features, cluster separation, checkpoints, TensorBoard logs, full prediction and original-test ID validation. No external business data or model weights are needed for the hybrid fixture.

The full original dataset and an NVIDIA GPU are not available in this development environment. The production multilingual model revision was checked against Hugging Face metadata. This does not constitute a production-model throughput/quality test. The CUDA-specific LightGBM test is skipped unless `RUN_CUDA_TESTS=1` on the target machine.

Before a long run, setup/doctor must pass in your SageMaker environment. Native-library memory estimates are heuristic; no OOM-free guarantee or improved leaderboard F0.5 is claimed.
