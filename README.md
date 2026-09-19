# Apex language-grounded 3D grasp pipeline

An end-to-end, stage-separated implementation scaffold derived from the supplied Stage 1, Stage 2, Stage 4, and whole-project architecture documents. The pipeline keeps language grounding, 2D object grounding, RGB-D spatial reasoning, and 6-DoF grasp proposal independently testable and emits explicit failure states rather than silently filling missing requirements.

Start with `docs/REQUIREMENTS_AND_DOWNLOADS.md`, then `MANUAL_RUN_GUIDE.md`. The project deliberately performs **no automatic model or dataset downloads** and leaves every hardware/model/calibration/benchmark decision explicit in `configs/project.required.yaml`.

Key entry points:

- `python -m apex_grasp.stage1_language.run`
- `python -m apex_grasp.stage2_object.run`
- `python -m apex_grasp.stage3_3d.run`
- `python -m apex_grasp.stage4_grasp.run`
- `python -m apex_grasp.run_pipeline`
- `python -m apex_grasp.eval.stage1|stage2|stage3|stage4|end_to_end`

`configs/document_baseline.example.yaml` contains only the numerical starting values explicitly suggested by the source specifications and labels them as non-optimal baselines.
