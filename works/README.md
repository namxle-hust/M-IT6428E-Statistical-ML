# M-IT6428E-Statistical-ML

```bash
python train.py configs/cascade-rcnn-hrnetv2p-w32-10epoch.py
```

```bash
!torchrun --nproc_per_node=2 /kaggle/input/parasiteegg/train.py /kaggle/input/parasiteegg/kaggle-cascade-rcnn-hrnetv2p-w32-10epoch.py --launcher pytorch
```

```bash
docker run -tiv .:/workspace namxle/python:test bash

python3 calculate_metrics.py
```

# load_from = "https://github.com/SwinTransformer/storage/releases/download/v1.0.2/cascade_mask_rcnn_swin_base_patch4_window7.pth"