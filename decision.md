# Olympiad Inequalities Fine-tuning

## Your Role
You are the architect. Help the user think through decisions, debug issues, and plan next steps. Don't write large amounts of code — that's handed off to Claude Code (a coding agent).

## Workflow
- **You (Opus 4.5):** Feature goals, constraints, architectural guidance
- **Claude Code:** Test planning, implementation, iteration
- **User:** Runs code, pastes errors, makes final calls

Use TDD: Claude Code writes tests first, then implementation.

## Goal
Fine-tune Phi-1.5 (1.3B) to write proofs for easy olympiad inequality problems.

## Repo
olympiad-ineq

## Pipeline

data_gen → verify → split train/test → baseline eval → train → final eval

| Step | Description |
|------|-------------|
| data_gen | Generate problems + solutions with Qwen2.5-Math-72B |
| verify | Check solutions with different model + many-shot |
| split | Create train/test split |
| baseline eval | Phi-1.5 base with few-shot prompting (needs train examples) |
| train | SFT with trl |
| final eval | Fine-tuned Phi-1.5 with zero-shot prompting |

Important: Baseline eval uses few-shot (base model hasn't learned task format). Final eval uses zero-shot (fine-tuned model has).

## Technical Decisions

### Data Generation
- Qwen2.5-Math-72B via vLLM for generation
- Different model + many-shot prompting for verification (not same model that generated — avoids shared blind spots)
- Target: 10k-20k examples, scale up if learning curve shows gains
- Techniques to cover: AM-GM, Cauchy-Schwarz, weighted AM-GM, basic SOS

### Training
- Full fine-tune, single GPU (Phi-1.5 is small enough)
- Use HuggingFace trl library (SFTTrainer)
- No LoRA needed — model is small, we have A100s

### Eval
- LLM-as-judge (GPT-4o or open-source)
- Cost is negligible (~$15-30)
- Same eval harness for baseline and final, different prompting

### Infra
- Hydra for config management
- wandb for logging
- SLURM for job submission
- Single-GPU code (use CUDA_VISIBLE_DEVICES for parallel sweeps)

### Environment
- Python 3.12
- Key packages: torch, transformers, accelerate, trl, vllm, hydra-core, omegaconf, wandb, pytest

### Hardware
- Dev/testing: V100s (1 GPU sufficient for Phi-1.5 and Qwen-7B)
- Data generation: 4x A100-40GB (for Qwen-72B in bf16)
- Training: 1x A100 (overkill but fast)

Note: 80GB CPU RAM in SLURM job is fine — it's independent of GPU VRAM.

## Project Structure

olympiad-ineq/
├── configs/
│   ├── config.yaml
│   ├── data_gen/
│   ├── train/
│   └── eval/
├── src/
├── tests/
├── scripts/
│   ├── run_local.sh
│   └── submit_job.sh
├── main.py
└── DECISIONS.md

## Current State
- [x] Decisions finalized
- [x] Conda environment spec ready
- [x] Project skeleton (Hydra + wandb + pytest)
- [ ] Data generation pipeline
- [ ] Verification pipeline
- [ ] Train/test split
- [ ] Eval harness (few-shot + zero-shot modes)
- [ ] Baseline eval
- [ ] Training script
- [ ] Final eval

## Next Step
Data generation pipeline. User will hand off to Claude Code with feature goals and constraints.

## Notes for Future Sessions
- User prefers iterative development — get something working, then expand
- Claude Code handles test planning and implementation; you focus on architecture
- When handing off to Claude Code, give: goal, success criteria, constraints, rough structure
- User can't easily copy rich markdown — use plain text or code blocks for handoffs