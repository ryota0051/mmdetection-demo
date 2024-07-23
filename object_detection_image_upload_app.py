import cv2
import mmcv
import numpy as np
import torch
import streamlit as st
from mmdet.apis import inference_detector, init_detector
from mmdet.registry import VISUALIZERS


MODEL_NAME = 'convnext'

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


SETTING_DICT = {
    'convnext': {
        'config': '/mmdetection/configs/convnext/mask-rcnn_convnext-t-p4-w7_fpn_amp-ms-crop-3x_coco.py',
        'weights': '/models/mask_rcnn_convnext-t_p4_w7_fpn_fp16_ms-crop_3x_coco_20220426_154953-050731f4.pth',
    },
    'swin': {
        'config': '/mmdetection/configs/swin/mask-rcnn_swin-t-p4-w7_fpn_ms-crop-3x_coco.py',
        'weights': '/models/mask_rcnn_swin-t-p4-w7_fpn_ms-crop-3x_coco_20210906_131725-bacf6f7b.pth'
    }
}


@st.cache_resource
def load_model(model_name):
    model = init_detector(SETTING_DICT[model_name]['config'], SETTING_DICT[model_name]['weights'], device=DEVICE)
    visualizer = VISUALIZERS.build(model.cfg.visualizer)
    visualizer.dataset_meta = model.dataset_meta
    return model, visualizer

choice = st.selectbox('モデルを選択してください', ['convnext', 'swin'], index=0)


model, visualizer = load_model(choice)


st.title("Object Finder")


file_path = st.file_uploader('', type=['png', 'jpg', 'jpeg'])

if file_path:
    image_bytes = file_path.read()
    img = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
    result = inference_detector(model, img)

    img = mmcv.imconvert(img, 'bgr', 'rgb')
    visualizer.add_datasample(
        name='result',
        image=img,
        data_sample=result,
        draw_gt=False,
        pred_score_thr=0.2,
        show=False
    )
    pred_img = visualizer.get_image()
    col1, col2 = st.columns(2)
    with col1:
        st.write('Base')
        st.image(img, channels='RGB')
    with col2:
        st.write('Detection result')
        st.image(pred_img, channels='RGB')
