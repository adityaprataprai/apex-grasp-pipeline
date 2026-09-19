# 1-Hour Submission Rescue Path

This is a **deterministic demo/replay path**, not the optimum benchmark path. It intentionally does not require Grounding DINO, SAM 2, CUDA Toolkit, C++ Build Tools, GraspNet, or external datasets. Those remain explicit external requirements in REQUIREMENTS_AND_DOWNLOADS.md.

## 1. Stop heavy installers
Cancel CUDA / Visual Studio / model installation attempts. They are not required for this replay demo.

## 2. In the activated .venv, install only the project packages
```
python -m pip install -r requirements-core.txt
python -m pip install -e .
```

## 3. Run the deterministic tests
```
python -m pytest -q
```

## 4. Run the full deterministic demo
```
python -m apex_grasp.run_pipeline --config configs/submission_demo.yaml --request-id submission_demo --scene-id synthetic01 --image data/demo/rgb.png --depth data/demo/depth.npy --intrinsics data/demo/intrinsics.json --instruction "Grab the blue carton behind the transparent bottle using its exposed side."
```

## 5. Expected artifacts
`artifacts/submission_demo/submission_demo.stage1.json`
`artifacts/submission_demo/submission_demo.stage2.json`
`artifacts/submission_demo/submission_demo.stage3.json`
`artifacts/submission_demo/submission_demo.stage4.json`
`artifacts/submission_demo/submission_demo.trace.json`

## 6. What this proves
It exercises the complete software contracts: language parsing, object candidate/replay grounding, RGB-D back-projection and BEHIND verification, grasp validation/scoring, and trace output. It does **not** prove Grounding DINO/SAM2/GraspNet benchmark performance.

## 7. Do not claim benchmark numbers from this demo
For benchmark claims, install the exact models/checkpoints/datasets and populate the real manifests as described in `docs/REQUIREMENTS_AND_DOWNLOADS.md`.
