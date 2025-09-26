Download the prepared data from the provided link: [Typ-Examples](https://1drv.ms/f/c/411e3f963c5f74e5/EpJ0BN4VwbBGsUr5-cleSvcBPPd3Nj0BMwXBt1vhu7qONw?e=hJZRGv).

The package contains three zip files:
1. data
2. outputs

After downloading, unzip the files into the `Shape-for-Motion/data` and `Shape-for-Motion/outputs` subfolders accordingly.

Toy examples:
1. Rotate by 20 degrees
```
bash scripts/s2_propagate_edit.sh otter-short demo scale20_rotate_r20 0
(Note: set mesh_rescaling_ratio=20)

bash scripts/s3_prepare_data.sh otter-short scale20_rotate_r20

bash scripts/s3_test.sh otter-short_scale20_rotate_r20
```

2. Modify the texture
```
bash scripts/s2_propagate_edit.sh dancing-patrick-star demo texture 0 
(Note: use canonical_w_color.ply)

bash scripts/s3_prepare_data.sh dancing-patrick-star texture

bash scripts/s3_test.sh dancing-patrick-star_texture
```

3. Add a new object
```
bash scripts/s2_propagate_edit.sh car-turn demo car_w_tree_98485 1

bash scripts/s3_prepare_data.sh car-turn car_w_tree_98485

bash scripts/s3_test.sh car-turn_car_w_tree_98485
```

4. pose editing
```
If you want to try pose editing, we suggest first building a bone structure and then deforming the 3D mesh based on it. You can follow the steps in this [tutorial](https://youtu.be/jj4IZ5iEzAo?si=HS_fk4sAS3LP5PMz) for detailed guidance. Other deformation methods are also possible, but using bones is generally the most intuitive and physics-aware approach. 
 
If you plan to perform auto-rigging, be cautious: external tools may alter or discard the vertex indices stored in the canonical mesh.
```