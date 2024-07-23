import av
import mmcv
import torch
import streamlit as st
from mmdet.apis import inference_detector, init_detector
from mmdet.registry import VISUALIZERS
from streamlit_webrtc import webrtc_streamer


MODEL_NAME = 'convnext'

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

TH = 0.5


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
    print(f'{model_name} used')
    model = init_detector(SETTING_DICT[model_name]['config'], SETTING_DICT[model_name]['weights'], device=DEVICE)
    visualizer = VISUALIZERS.build(model.cfg.visualizer)
    visualizer.dataset_meta = model.dataset_meta
    return model, visualizer


st.title('Object Detection Example')

choice = st.selectbox('モデルを選択してください', ['convnext', 'swin'], index=0)




def od_callback(frame):
    model, visualizer = load_model(choice)
    img = frame.to_ndarray(format='bgr24')

    result = inference_detector(model, img)

    img = mmcv.imconvert(img, 'bgr', 'rgb')
    visualizer.add_datasample(
        name='result',
        image=img,
        data_sample=result,
        draw_gt=False,
        pred_score_thr=TH,
        show=False
    )
    pred_img = visualizer.get_image()

    return av.VideoFrame.from_ndarray(pred_img, format='rgb24')


webrtc_streamer(key='object_detection', video_frame_callback=od_callback)
