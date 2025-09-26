Refer to the following data directory structure for data construction

```
├── DATA.md
├── raw_videos
│   └── <name>.mp4
├── s1_processed
│   └── <name>
│       ├── novelviews
│       ├── processed_depths
│       ├── processed_masks
│       ├── processed_rgbs
│       ├── raw-depths
│       ├── raw-masks
│       ├── raw-rgbs
│       ├── <name>.json
│       ├── points3d.ply
│       └── transform_params.pkl
└── s2_edited
    ├── edited_videos
    │   └── <name>
    │       ├── edited-mask
    │       │   └── <name>.mp4
    │       ├── edited-normal
    │       │   └── <name>.mp4
    │       ├── edited-rgb
    │       │   └── <name>.mp4
    │       ├── excluded-mask
    │       │   └── <name>.mp4
    │       ├── ori-mask
    │       │   └── <name>.mp4
    │       └── ori-rgb
    │           └── <name>.mp4
    └── editing_files
        └── <name>
            ├── canonical_w_color.ply
            ├── canonical_wo_color.ply
            └── edited_mesh.ply
```

- **raw_videos**: Stores input videos.

- **s1_processed**: Outputs from Stage 1 processing (per-sequence folders), including:
  - **\<name>.json**: Camera intrinsics/extrinsics and related metadata.
  - **points3d.ply**: sphere points initialization
  - **novel_views**: Rendered/novel viewpoints.
  - **processed_depths/masks/rgbs**: prepared inputs after the center-cropping
  - **raw-depth/masks/rgbs**: Estimated depth maps, masks and the original rgbs
  - **transform_params.pkl**: image crop的coordinate

- **s2_edited**: Inputs/outputs for editing and preparation for Stage 3 rendering.
  - **editing_files**: 
    - **canonical_w/wo_color.ply**: the canonical mesh reconstructed in Stage 1
    - **edited_mesh.ply**: the user edited mesh, e.g., rotation_left_20.ply, scaling.ply, and etc.
  - **edited_videos**: For each matching `<name>`, contains paired original and edited video frames and masks used by Stage 3 generative rendering:
    - **edited-mask**: Binary masks of edited regions.
    - **edited-normal**: Surface normals of edited content (if applicable).
    - **edited-rgb**: RGB frames after editing.
    - **excluded-mask**: Masks for regions excluded from editing.
    - **ori-mask**: Original masks before editing.
    - **ori-rgb**: Original RGB frames.

