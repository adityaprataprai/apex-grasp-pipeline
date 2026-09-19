# Manual run guide (no Bash scripts)

This guide is written for manually entering commands in **Windows PowerShell or Command Prompt**. There are no `.sh`/Bash setup or run scripts in this project. The project will not download checkpoints, datasets, or model repositories automatically.

## A. Prepare the environment

1. Read `docs/REQUIREMENTS_AND_DOWNLOADS.md` completely first. Decide which exact parser, Grounding DINO implementation/checkpoint, SAM 2 implementation/checkpoint, attribute verifier, 6-DoF grasp generator, collision/simulation backend, dataset versions, camera and gripper you will use.
2. Install Python 3.10+ and Git. Install your chosen GPU driver/CUDA/PyTorch combination according to the official PyTorch/model-repository instructions. The supplied documents do not choose these versions.
3. Open PowerShell or Command Prompt in the project folder.
4. Create a virtual environment: `python -m venv .venv`
5. Activate it in PowerShell with `.venv\Scripts\Activate.ps1`, or in Command Prompt with `.venv\Scripts\activate.bat`.
6. Install this project: `python -m pip install -e .`
7. For testing/evaluation: `python -m pip install -r requirements-eval.txt`
8. For Open3D/trimesh helpers: `python -m pip install -e ".[3d]"`
9. Separately clone/install the exact official Grounding DINO and SAM 2 versions you decided to use, then download their matching checkpoints. Separately install/connect your chosen GraspNet-family generator. Record commit hashes/checkpoint filenames in your experiment log.

## B. Create a real configuration

Copy `configs/project.required.yaml` to a new file such as `configs/local.yaml`. Fill **every field required by the execution path you selected**. `null` means deliberately unset; the runner will fail instead of guessing.

You may inspect `configs/document_baseline.example.yaml` for values explicitly suggested by the documents, but do not copy them as if they were optimum. Validate/tune them on a held-out validation split.

## C. Prepare one scene

For one end-to-end RGB-D replay, prepare:

- `scene_rgb.png` — RGB image.
- `scene_depth.npy` or a depth image — aligned with RGB.
- `intrinsics.json` — e.g. `{"fx": ..., "fy": ..., "cx": ..., "cy": ...}` from your calibrated camera.
- optionally `camera_to_scene.json` — 4x4 transform if scene axes differ from camera axes.
- the natural-language instruction.

Depth values are multiplied by `stage3.depth_unit_to_m`, so that field must exactly match your depth representation.

## D. Run stages individually

Stage 1:
`python -m apex_grasp.stage1_language.run --config configs/local.yaml --instruction "Grab the semi-crushed parcel behind the transparent bottle using its exposed side." --output artifacts/stage1.json`

Stage 2:
`python -m apex_grasp.stage2_object.run --config configs/local.yaml --image scene_rgb.png --task-ir artifacts/stage1.json --output artifacts/stage2.json`

Stage 3:
`python -m apex_grasp.stage3_3d.run --config configs/local.yaml --task-ir artifacts/stage1.json --stage2 artifacts/stage2.json --depth scene_depth.npy --intrinsics intrinsics.json --output artifacts/stage3.json`

If you have a camera-to-scene transform, add `--camera-to-scene camera_to_scene.json`. Stage 3 also writes `artifacts/stage3.json.target_points.npy`.

For Stage 4 you need the observed scene-obstacle cloud excluding the target. The end-to-end runner builds this automatically. If running Stage 4 alone, provide your own `scene_obstacles.npy` in the same scene frame:
`python -m apex_grasp.stage4_grasp.run --config configs/local.yaml --task-ir artifacts/stage1.json --target artifacts/stage3.json --target-points artifacts/stage3.json.target_points.npy --scene-points scene_obstacles.npy --output artifacts/stage4.json`

## E. Run the full pipeline

`python -m apex_grasp.run_pipeline --config configs/local.yaml --request-id demo001 --scene-id scene001 --image scene_rgb.png --depth scene_depth.npy --intrinsics intrinsics.json --instruction "Grab the semi-crushed parcel behind the transparent bottle using its exposed side."`

Add `--camera-to-scene camera_to_scene.json` when required by your coordinate setup. Intermediate artifacts and failure JSON are written to the configured artifact directory.

## F. Run evaluations

Each evaluator consumes JSONL rows. Templates live in `data/manifests/`.

Stage 1: `python -m apex_grasp.eval.stage1 --manifest data/manifests/stage1_eval.template.jsonl --output artifacts/stage1_metrics.json`

Stage 2: `python -m apex_grasp.eval.stage2 --manifest data/manifests/stage2_eval.template.jsonl --output artifacts/stage2_metrics.json`

Stage 3: `python -m apex_grasp.eval.stage3 --manifest data/manifests/stage3_eval.template.jsonl --output artifacts/stage3_metrics.json`

Stage 4: `python -m apex_grasp.eval.stage4 --manifest data/manifests/stage4_eval.template.jsonl --output artifacts/stage4_metrics.json`

End-to-end: `python -m apex_grasp.eval.end_to_end --manifest data/manifests/e2e_eval.template.jsonl --output artifacts/e2e_metrics.json`

The template rows are examples of field shape, not benchmark data. Replace them with real annotations/prediction paths.

## G. Run unit tests

`python -m pytest -q`

The included tests cover schema validation, projection geometry, relation logic helpers, scoring/collision basics, and deterministic contract behavior. They do not replace the dataset benchmarks.

## H. Before reporting “optimum” results

Freeze and report: repository commits; model/checkpoint IDs; all config values; hardware/driver/PyTorch versions; dataset versions and split IDs; camera/depth calibration; gripper/robot geometry; random seed; warm-up protocol; latency measurement protocol; whether evaluation used RGB-D or a monocular fallback; collision/simulator backend; and whether any custom scenes used for tuning also appear in test. Do not tune on the final test split.

## I. Optional deterministic synthetic geometry fixture

For coordinate/relation regression tests (not for claiming perception benchmark quality), you can generate a simple RGB-D scene from an explicit JSON spec:

`python -m apex_grasp.synthetic.generate --spec data/fixtures/synthetic_scene_spec.example.json --output-dir artifacts/synthetic_scene`

This creates RGB, metric depth, masks and annotations. It is deliberately simple and does not replace OCID, RefCOCO-family, GraspNet/TransCG, or your custom warehouse evaluation set.
