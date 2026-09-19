# Requirements and downloads before PC execution

This is the exhaustive preparation checklist implied by the supplied architecture/specifications. Nothing below is auto-downloaded by the project. Any item marked **benchmark-required** is needed to measure the corresponding acceptance target; an item marked **runtime-required** is needed to run that path at all. Items marked **optional** are only needed for the feature they name.

## 1. Computer and operating environment

**Runtime-required:** 64-bit PC, Python 3.10+ environment, enough local storage for the particular model checkpoints and datasets you choose, and permission to compile/install the native dependencies required by your chosen PyTorch/Open3D/model stack. The source documents do not specify a particular Windows/Linux version, CUDA version, GPU model, VRAM amount, RAM amount, or disk size. Those must be selected from the official requirements of the exact Grounding DINO, SAM 2, depth, GraspNet, simulator, and PyTorch versions you choose. Do not treat an arbitrary internet hardware recommendation as a project benchmark requirement.

**Strongly expected for the intended pretrained-model path:** a CUDA-capable NVIDIA GPU for practical Grounding DINO/SAM 2/GraspNet latency. CPU mode is technically possible for some components but should not be assumed to meet the Stage 1/2 latency targets. Benchmark on the actual target hardware.

## 2. Core Python packages

Install the repository itself plus core dependencies: NumPy, Pydantic, PyYAML, Pillow, and OpenCV. For 3D geometry install Open3D; for offline mesh/collision extensions install trimesh or an FCL-like library. For tests install pytest. These mirror the technologies named in the specifications.

The repository deliberately does not pin PyTorch/CUDA because the correct wheel depends on your GPU driver/CUDA choice and on the official model repositories you select. Install PyTorch according to the official matrix for your machine **before** model repositories.

## 3. Stage 1 language grounding

Choose exactly one parser backend before running:

- `openai_compatible`: you supply endpoint URL, model name, API-key environment-variable name, and timeout.
- `python_callable`: you supply a local parser implementation using the adapter contract.
- `rule`: included only as a deterministic sanity-test fallback; it is deliberately narrow and should not be claimed as the optimum language benchmark implementation.

**Benchmark-required data:** a hand-authored challenge instruction set of roughly 300-500 instructions, 50-100 hard-negative/ambiguity instructions, and optionally 1,000+ synthetic paraphrase/edge-case variants. RefCOCO-family referring-expression resources are useful background/sanity data. Split by scenario/template family, not by randomly mixing near-identical paraphrases.

## 4. Stage 2 object grounding

**Runtime-required for the baseline named by the spec:**

1. Grounding DINO official code/package.
2. A **versioned Grounding DINO model checkpoint** and any matching model config file required by that implementation.
3. SAM 2 official code/package.
4. A **versioned SAM 2 checkpoint** and its matching model configuration.
5. PyTorch compatible with those exact repositories/checkpoints.
6. If semantic conditions such as crushed/dented/reflective are part of your benchmark, an explicit attribute verifier: chosen VLM/classifier and checkpoint/API, or a dedicated trained model. The documents say these attributes may require a VLM/dedicated classifier; this repo does not invent one.
7. If `basic_color` is used, camera-specific calibrated HSV prototypes in a YAML file. The provided prototype file is intentionally empty.

**Benchmark-required/strongly recommended data:** OCID public benchmark split for clutter/segmentation; RefCOCO/RefCOCO+/RefCOCOg validation/test splits for referring-expression sanity; 100-200 custom warehouse scenes; 200+ hard-negative queries with similar objects; 50+ transparency/reflection scenes. GraspNet/TransCG RGB-D can also be used for cross-stage stress testing.

The document baseline mentions initial thresholds `box_threshold=0.30`, `text_threshold=0.25`, `MIN_ACCEPT_SCORE=0.60`, `MIN_MARGIN=0.10` and Stage 2 weights 0.30/0.20/0.20/0.10/0.20. They are starting values only, not optimum settings; tune and version them on a held-out validation set.

## 5. Stage 3 3D reasoning

**Runtime-required:** aligned depth for the RGB frame (RGB-D is the preferred first implementation), camera intrinsics `fx, fy, cx, cy`, correct depth-unit conversion to metres, and masks from Stage 2. If you need world/robot-axis relations such as ABOVE with a tilted camera, supply a calibrated 4x4 camera-to-scene transform and a scene up-axis. If you omit a transform, the implementation stays in the camera frame; you must decide whether that is valid for your experiment.

**Configuration that must be measured/calibrated rather than guessed:** valid depth range, minimum object point count, outlier percentiles, voxel size, BEHIND depth margin, required image-plane overlap, NEAR distance, BETWEEN corridor size, WEDGED clearance, accessibility probe distance, relation-score threshold, final target score threshold, final target score margin, and Stage 3 evidence weights.

**Benchmark-required/strongly recommended data:** OCID RGB-D for clutter/3D tests plus custom RGB-D scenes with ground-truth target centroids/relations/accessibility. The whole-project document defines Stage 3 metrics but does not provide numeric acceptance thresholds; establish them from sensor accuracy and your held-out validation set rather than inventing universal values.

**RGB-only fallback:** the architecture mentions a controlled monocular-depth fallback, but also explicitly recommends implementing RGB-D first. A monocular depth model/checkpoint is therefore not selected or bundled here. If you need RGB-only operation, add the exact model, calibration/scaling method, validation dataset, and acceptance threshold as an explicit project decision.

## 6. Stage 4 grasp proposal

**Runtime-required:** a chosen 6-DoF grasp generator (the spec names the GraspNet family as the primary baseline), its code and checkpoint, and an adapter conforming to `docs/ADAPTER_CONTRACTS.md`; actual gripper geometry and width limits; robot workspace limits; scene/obstacle point cloud; target point cloud; a verified target; and all safety margins/approach sampling parameters.

**Benchmark-required:** GraspNet-1Billion is the main external grasp benchmark named by the spec. The spec states 190 cluttered scenes, 97,280 images, 88 objects, and dense 6D grasp labels. TransCG/transparent-object resources are recommended for the transparent-bottle challenge. A custom dataset should emphasize partial occlusion, constrained grasp regions, narrow gaps, bin-wall proximity, cables, reflective/transparent objects, and should annotate target masks, valid grasps, invalid-grasp reasons, approach clearance, and instruction constraints.

**Collision/simulation options:** Open3D is used for point clouds. The included validator is an offline point-cloud gripper-volume proxy so the project has deterministic unit/integration behavior. For optimum robot-faithful collision benchmarks, select and install one explicit geometry/robot collision stack: trimesh/FCL-like collision for deterministic offline mesh tests, PyBullet for lightweight simulation, Isaac Sim for higher-fidelity NVIDIA simulation, or MoveIt 2 with a robot model for robot-aware collision/motion planning. Do not combine results from these backends without naming the backend and configuration.

The document baseline suggests generating top K=100 raw grasps offline and Stage 4 ranking weights 0.35/0.20/0.15/0.10/0.10/0.10. These are starting values only. The real gripper dimensions, workspace, collision margin, safe approach distance and minimum contact support cannot be inferred from the documents and must be supplied from your hardware/benchmark.

## 7. Robot integration, only if executing on hardware

**Optional for the offline challenge demo, required for real execution:** ROS 2 if you choose it as middleware, MoveIt 2 or equivalent motion planner, URDF/SRDF and collision meshes for the exact robot/gripper, calibrated camera-to-robot transforms, inverse-kinematics solver/limits, controller interface, hardware emergency-stop/safety configuration, and robot-specific execution tests. The architecture puts robot execution behind the validated-grasp boundary; the language model must not directly drive the robot.

## 8. Files you must create before a real run

1. A filled configuration copied from `configs/project.required.yaml`.
2. Camera intrinsics JSON: `{"fx": ..., "fy": ..., "cx": ..., "cy": ...}`.
3. Optional camera-to-scene 4x4 JSON if scene/world axes differ from the camera frame.
4. RGB image and aligned depth image/NumPy array for each scene.
5. Stage-specific dataset manifests if you want quantitative evaluation.
6. Exact model/checkpoint paths and model-version labels.
7. Color prototypes if using basic color verification.
8. Gripper dimensions and robot workspace limits.
9. An explicit Stage 4 generator adapter or replay file.

## 9. Acceptance targets copied from the supplied specifications

Stage 1: JSON validity >=99%; action accuracy >=98%; target category F1 >=95%; attribute F1 >=92%; relation accuracy >=95%; ambiguity F1 >=90%; hallucination rate <=1%; P95 parse latency <=2 s on target hardware.

Stage 2: Grounding Accuracy@1 >=90%; Recall@5 >=97%; mean box IoU >=0.70; mean mask IoU >=0.75; supported-attribute accuracy >=90%; ambiguity deferral >=90% on intentionally ambiguous cases; ECE <=0.10 initial target; P95 latency <=5 s initial target.

Stage 4: top-1 grasp validity >=90% offline; top-10 valid-grasp recall >=95%; post-filter collision-free rate >=95%; Stage 1 constraint satisfaction >=95% on constrained subset; accepted-grasp clearance above the configured safety margin; simulator grasp success >=80% initial target. Report pre-filter and post-filter results together.

Stage 3 and end-to-end: the architecture names required metrics but does not give numeric universal thresholds. This project intentionally leaves them unset.
