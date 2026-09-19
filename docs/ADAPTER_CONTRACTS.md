# Adapter contracts

The specifications intentionally leave several model choices replaceable. This repository therefore does **not** choose a language model, VLM, GraspNet fork, checkpoint, robot, or gripper for you. The following contracts let you connect the exact implementation you decide to benchmark.

## Stage 1 `python_callable`
Configure `stage1.python_callable.module` and `function`. The function is called as `fn(text: str)` and must return a dictionary that validates as `TaskIR` in `src/apex_grasp/common/models.py`.

## Stage 2 detector `python_callable`
Called as `fn(image: np.ndarray RGB uint8, prompts: list[str])`. Return a list of dictionaries containing at minimum:

`{"bbox_xyxy": [x1,y1,x2,y2], "category_score": float, "prompt_score": float, "prompt": str, "raw_id": str}`

Scores must be normalized to `[0,1]` before they enter the shared ranking code.

## Stage 2 segmenter `python_callable`
Called as `fn(image, boxes_xyxy)`. Return one boolean/0-1 HxW NumPy mask per input box, in the same order.

## Stage 2 attribute verifier `python_callable`
Called as `fn(image, mask, requested_attributes)`. Return `(score, metadata)` where `score` is in `[0,1]`. If you convert a discrete VLM answer to a probability, calibrate that mapping on held-out validation data first.

## Stage 4 grasp generator `python_callable`
Called with keyword arguments:

`fn(target_points=<Nx3 metres>, scene_points=<Mx3 metres>, target_object_id=<str>, context=<dict>)`

Return a list of dictionaries that validate as `GraspCandidate`. Each candidate needs: `target_object_id`, `position_xyz` in metres, `quaternion_xyzw`, normalized `approach_vector`, `jaw_width_m`, `model_quality` in `[0,1]`. Put a stable model-side identifier into `provenance.grasp_id` if you want exact top-K benchmark recall.

### Coordinate convention requirement
The Stage 4 geometric proxy assumes the grasp generator and collision validator use the same scene frame. The included proxy interprets local gripper X as jaw-opening axis and local Z as the gripper depth/approach-body axis. If your generator uses another convention, write an adapter that converts poses before returning them. Do not benchmark before verifying this with known-pose fixtures.

## Replay backends
Replay adapters are provided so pipeline contracts and evaluation can be tested without loading large models. They are not benchmark substitutes. A replay JSON can contain `detections` (Stage 2) or `grasps` (Stage 4) in the schemas above.
