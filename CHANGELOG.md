# Changelog

## 2.0.0 (2026-10)

The Tripo v2 API is shut down on 2026-10-31. This release runs every node on Tripo API v3 through
`tripo3d>=1.0.0,<2.0.0`.

### Compatibility

- Requires `tripo3d>=1.0.0,<2.0.0` (pinned in both `requirements.txt` and `pyproject.toml`).
- Saved workflows keep loading: no input name, input order or existing option was changed or removed;
  new inputs were added at the end with defaults that keep the old behaviour. No node was removed.
- If an incompatible `tripo3d` is installed, ComfyUI still starts; the log and every Tripo node report
  the `pip install "tripo3d>=1.0.0,<2.0.0"` command to run.
- 1.3.x of this extension also keeps working on `tripo3d` 1.x until you update it.

### Added

- **Tripo: Smart UV** (`TripoMeshUV`): UV unwrap with a UV atlas `IMAGE` output.
- Generate model: `P1-20260311`, `P2-20260801`; `export_orientation`.
- Texture model: `v3.5-20260815`, `texture_quality=fast`, `delight` (auto / on / off).
- Rig model: `v2.5-20260210`, `v3.0-20260909`; specs `actorcore`, `unity`, `unreal`, `vrm`;
  optional `rig_type` (auto runs the rig check first, as before).
- Smart low poly: `P-v3.0-20260909` and `smart_uv`; `face_limit <= 0` uses the server default.
- Complete mesh: `completion_mode` (`ai_completion`, `quick_cap`).
- Convert model: `fbx_preset=bake_scale`.

### Changed

- Import model sends `model_url` to the API instead of downloading it inside ComfyUI first.
- Rig model `v2.0-20250506` runs as `v2.5-20260210` (same as API v2). v2.5 rigs only accept animal
  presets; new Rig nodes default to `v1.0-20240301`, which accepts the humanoid presets.
- `bake_animation` on Retarget is kept for saved workflows but ignored by API v3.

### Deprecated

- **Tripo: Refine Draft model** (`TripoRefineModel`) and **Tripo: Stylize model** (`TripoStylizeModel`):
  Tripo no longer supports refine or stylize. The nodes are marked deprecated (hidden from the node search)
  and kept with unchanged inputs so saved workflows still open; running them stops immediately with
  "no longer supported … remove the node" — no request is sent and no credits are used.
