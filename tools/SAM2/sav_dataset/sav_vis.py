from megfile import smart_path_join, smart_open, smart_exists
from utils.sav_utils import SAVDataset
from PIL import Image

# data_root='s3+my_s_hdd_new://pbr-obj/alpha'
# aws --profile aigc --endpoint-url=http://p-ceph-norm-outside.pjlab.org.cn s3 ls s3://public-dataset-p2/en-public-sam-video/sav_train/sav_000/
data_root = 's3+AIGC_GENERAL://public-dataset-p2/en-public-sam-video/sav_train/sav_000'
# rgba = Image.open(smart_open(image_file, 'rb'))

sav_dataset = SAVDataset(sav_dir=data_root)
frames, manual_annot, auto_annot = sav_dataset.get_frames_and_annotations("sav_000001")

sav_dataset.visualize_annotation(
    frames, manual_annot, auto_annot,
    annotated_frame_id=0,
    show_auto=False,
)


"""
1. megfile_env AIGC_GENERAL FB7QKWTWP279SQMLBX4H dN6ph2f9cQcVhnOCngiGKwPUjMqpM9o4oiKM67mb http://p-ceph-norm-outside.pjlab.org.cn
2. data_root = 's3+AIGC_GENERAL://xxx'

3. sav_test: 150; sav_val: 155
"""