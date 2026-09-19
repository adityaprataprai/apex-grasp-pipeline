# Apex: Language-Grounded 3D Grasp Pipeline

An end-to-end robotics perception pipeline that turns a natural-language instruction into a validated 6-DoF grasp proposal.

A simple example is:

> **“Grab the blue carton behind the transparent bottle using its exposed side.”**

The system breaks that request into four independently testable stages:

```text
Natural-language instruction
            │
            ▼
   Stage 1: Language Grounding
   What object? What relation?
            │
            ▼
   Stage 2: Object Grounding
   Where is the object in the image?
            │
            ▼
   Stage 3: 3D Scene Reasoning
   Does the RGB-D scene satisfy the relation?
            │
            ▼
   Stage 4: Grasp Generation
   Which 6-DoF grasp is valid and safe?
            │
            ▼
       Final grasp proposal
```

The project is designed around explicit stage interfaces, machine-readable artifacts, deterministic failure states, and reproducible evaluation. It does **not** silently choose missing hardware, calibration, model checkpoints, or benchmark settings.

---

## What this repository contains

The repository contains the complete software scaffold for the four-stage pipeline:

| Stage | Purpose | Main output |
|---|---|---|
| **1. Language grounding** | Converts an instruction into structured task information such as action, target object, attributes, and spatial relation | `stage1.json` |
| **2. Object grounding** | Finds the requested object in the RGB image and associates candidate masks/boxes with the language description | `stage2.json` |
| **3. 3D reasoning** | Uses RGB-D geometry and camera calibration to verify spatial relations such as `BEHIND`, `ABOVE`, `NEAR`, and `BETWEEN` | `stage3.json` |
| **4. Grasp generation** | Generates, validates, scores, and ranks 6-DoF grasp proposals | `stage4.json` |

The repository also includes:

- stage-specific evaluation programs
- end-to-end evaluation support
- configuration templates
- adapter contracts for external models
- deterministic tests
- a small synthetic RGB-D demonstration
- generated stage artifacts from the submission demo

---

## Quick start: run the included demo

The fastest way to see the project working is the included deterministic demo. It is designed to run without downloading large perception models or external datasets.

### 1. Requirements

You need:

- Python **3.10 or newer**
- Git
- Windows, Linux, or macOS
- a terminal such as Command Prompt, PowerShell, or a normal shell

The included demo does **not** require Grounding DINO, SAM 2, GraspNet, CUDA Toolkit, or a robot.

### 2. Clone the repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd apex-grasp-pipeline
```

### 3. Create a virtual environment

Windows Command Prompt:

```bat
python -m venv .venv
.venv\Scripts\activate
```

Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

Linux/macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 4. Install the project dependencies

```bash
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements-core.txt
python -m pip install -r requirements-eval.txt
python -m pip install -e .
```

### 5. Run the tests

```bash
python -m pytest -q
```

The included test suite checks the core schemas, geometry/relation logic, scoring, and deterministic pipeline contracts.

### 6. Run the complete demo

```bash
python -m apex_grasp.run_pipeline --config configs/submission_demo.yaml --request-id submission_demo --scene-id synthetic01 --image data/demo/rgb.png --depth data/demo/depth.npy --intrinsics data/demo/intrinsics.json --instruction "Grab the blue carton behind the transparent bottle using its exposed side."
```

A successful run prints:

```text
SUCCESS: submission_demo
```

### 7. Inspect the generated results

The pipeline writes its outputs to:

```text
artifacts/submission_demo/
├── submission_demo.stage1.json
├── submission_demo.stage2.json
├── submission_demo.stage3.json
├── submission_demo.stage4.json
├── submission_demo.trace.json
├── submission_demo.scene_obstacles.npy
└── submission_demo.target_points.npy
```

These files provide a machine-readable record of the four stages and the end-to-end execution trace.

---

## How the full system works

### Stage 1 — Language grounding

The first stage interprets the natural-language task and produces a structured representation. Depending on the selected backend, this can be backed by an OpenAI-compatible endpoint, a local Python callable, or the narrow deterministic rule backend used for sanity tests.

Typical information includes:

- action, such as `GRASP`
- target category
- target attributes
- spatial relation
- ambiguity information
- structured task constraints

### Stage 2 — Object grounding

The second stage connects the language description to image regions. The intended pretrained-model path uses **Grounding DINO** for language-conditioned detection and **SAM 2** for segmentation, with an optional attribute-verification component where the benchmark requires it.

The project keeps these external models behind an adapter interface so that model versions and checkpoints remain explicit.

### Stage 3 — 3D reasoning

The third stage combines the Stage 2 masks with RGB-D information and camera calibration.

It can reason about spatial relationships using 3D points rather than relying only on 2D image positions. For example, a `BEHIND` relation can be checked from the reconstructed object geometry and the configured scene/camera frame.

Required inputs for a real RGB-D run include:

- RGB image
- aligned depth
- camera intrinsics
- correct depth-unit conversion
- optional camera-to-scene transform

### Stage 4 — grasp generation

The final stage takes the verified target geometry and candidate scene obstacles and produces 6-DoF grasp proposals.

For the intended full benchmark path, the specification identifies the **GraspNet family** as the primary external baseline. The repository also includes a deterministic point-cloud gripper-volume validation path so that the software contracts can be tested without installing a full grasping stack.

---

## Running the individual stages

The stages can also be executed independently.

### Stage 1

```bash
python -m apex_grasp.stage1_language.run --config configs/local.yaml --instruction "Grab the semi-crushed parcel behind the transparent bottle using its exposed side." --output artifacts/stage1.json
```

### Stage 2

```bash
python -m apex_grasp.stage2_object.run --config configs/local.yaml --image scene_rgb.png --task-ir artifacts/stage1.json --output artifacts/stage2.json
```

### Stage 3

```bash
python -m apex_grasp.stage3_3d.run --config configs/local.yaml --task-ir artifacts/stage1.json --stage2 artifacts/stage2.json --depth scene_depth.npy --intrinsics intrinsics.json --output artifacts/stage3.json
```

If your scene requires a separate scene/world coordinate frame, also provide:

```text
--camera-to-scene camera_to_scene.json
```

### Stage 4

```bash
python -m apex_grasp.stage4_grasp.run --config configs/local.yaml --task-ir artifacts/stage1.json --target artifacts/stage3.json --target-points artifacts/stage3.json.target_points.npy --scene-points scene_obstacles.npy --output artifacts/stage4.json
```

The full end-to-end command is preferred when you want all intermediate artifacts to be generated automatically.

---

## Running the full real-model version

The included demo is intentionally lightweight. To run the research/benchmark configuration with the intended pretrained models, additional components must be installed separately.

The repository deliberately does **not** automatically download these dependencies.

You will need to select and document compatible versions of:

### Language grounding

Choose one parser backend:

- OpenAI-compatible API endpoint
- local Python callable
- rule backend for deterministic sanity checks only

### Object grounding

For the intended pretrained baseline:

- Grounding DINO code/package
- matching Grounding DINO checkpoint and model configuration
- SAM 2 code/package
- matching SAM 2 checkpoint and configuration
- compatible PyTorch installation
- optional attribute verifier for conditions such as `crushed`, `dented`, or `reflective`

### 3D reasoning

You need:

- aligned RGB-D data
- calibrated camera intrinsics
- correct depth scaling
- a calibrated camera-to-scene transform when required by your coordinate system
- measured/calibrated relation thresholds

### Grasp generation

You need:

- a selected 6-DoF grasp generator
- its code and checkpoint
- target point cloud
- scene obstacle point cloud
- actual gripper dimensions
- robot workspace limits
- collision/safety settings

The complete preparation checklist is in:

```text
docs/REQUIREMENTS_AND_DOWNLOADS.md
```

The detailed manual execution guide is in:

```text
MANUAL_RUN_GUIDE.md
```

---

## Configuration

The main configuration template is:

```text
configs/project.required.yaml
```

Copy it before creating a real local configuration, for example:

```text
configs/local.yaml
```

The project is intentionally strict about missing configuration. Values such as camera calibration, gripper geometry, collision margins, model checkpoints, and benchmark settings must be supplied explicitly rather than guessed.

`configs/document_baseline.example.yaml` contains numerical **starting values referenced by the project specifications**. They are baselines, not guaranteed optimum settings.

---

## Evaluation

Evaluation programs are provided for all four stages and for the complete pipeline:

```bash
python -m apex_grasp.eval.stage1
python -m apex_grasp.eval.stage2
python -m apex_grasp.eval.stage3
python -m apex_grasp.eval.stage4
python -m apex_grasp.eval.end_to_end
```

Each evaluator consumes a JSONL manifest. Example manifest templates are in:

```text
data/manifests/
```

The included templates demonstrate the expected data format. They are **not benchmark datasets**.

For real benchmark numbers, provide the appropriate datasets, annotations, model checkpoints, hardware information, calibration, and exact configuration used for the experiment.

---

## Datasets and external models

This repository does not bundle large third-party datasets or proprietary/model checkpoints.

The project documentation identifies resources such as:

- OCID for clutter/object grounding and RGB-D evaluation
- RefCOCO-family datasets for referring-expression evaluation
- GraspNet-1Billion for 6D grasp evaluation
- TransCG and transparent-object resources for transparent-object stress testing
- custom warehouse/RGB-D scenes for task-specific evaluation

Always follow the license and download instructions of the corresponding dataset/model repository.

---

## What the included demo proves

The deterministic demo proves that the **software pipeline and stage contracts can execute end-to-end** on the supplied synthetic/replay scene.

It exercises:

- language parsing
- object candidate/replay grounding
- RGB-D projection and spatial-relation verification
- grasp validation/scoring
- end-to-end artifact and trace generation

It does **not** establish benchmark performance for Grounding DINO, SAM 2, GraspNet, or any other external model. Benchmark claims require the actual models, checkpoints, datasets, and evaluation protocol to be run.

---

## Repository structure

```text
apex-grasp-pipeline/
│
├── src/apex_grasp/             # Python implementation
│   ├── stage1_language/        # Language grounding
│   ├── stage2_object/          # Object grounding
│   ├── stage3_3d/               # RGB-D 3D reasoning
│   ├── stage4_grasp/           # Grasp generation/validation
│   └── eval/                   # Evaluation entry points
│
├── configs/                    # Project and demo configurations
├── data/
│   ├── demo/                   # Small deterministic demo scene
│   ├── fixtures/               # Synthetic test fixtures
│   └── manifests/              # Evaluation manifest templates
│
├── docs/
│   ├── ADAPTER_CONTRACTS.md
│   └── REQUIREMENTS_AND_DOWNLOADS.md
│
├── tests/                      # Automated tests
├── artifacts/                  # Generated outputs
├── MANUAL_RUN_GUIDE.md         # Detailed manual execution guide
├── SUBMISSION_QUICKSTART_1H.md # Fast deterministic demo guide
├── requirements-core.txt
├── requirements-eval.txt
└── pyproject.toml
```

---

## Reproducibility

For a meaningful experiment, record:

- Git commit or repository version
- model and checkpoint versions
- configuration file
- Python/PyTorch versions
- GPU and driver information
- dataset versions and split IDs
- camera calibration
- gripper geometry
- random seed
- collision backend
- latency measurement procedure

Do not tune parameters on the final test set.

---

## Current scope

The repository focuses on the **perception → reasoning → validated grasp proposal** boundary. Hardware execution is intentionally separated from validated grasp generation.

For physical robot deployment, the project documentation calls for an explicit robot/camera calibration, robot model, motion planner, inverse kinematics, controller interface, and safety configuration rather than assuming a particular robot.

---

## License and external components

This repository contains project code and configuration plus references/adapters for external software and datasets. Third-party components remain subject to their respective licenses and terms.

Before publishing a public repository, verify that every external file, checkpoint, dataset sample, and generated artifact you commit is legally redistributable.

---

## Contact / project information

Add your team name, authors, institution, competition/challenge name, and contact information here before publishing the repository publicly.
