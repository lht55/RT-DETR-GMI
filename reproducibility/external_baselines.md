**YOLOv8n**
Framework: Ultralytics
Package version: 8.2.50
Official release/tag: v8.2.50
Commit: 26a664f636e3de5c1afce0f55c9742e3b4d931ac
Model: YOLOv8n
Model configuration: ultralytics/cfg/models/v8/yolov8.yaml, scale=n
Training model entry: ultralytics/cfg/models/v8/yolov8n.yaml
Dataset: /split_4481/data.yaml
Seeds: 0, 42, 3407
Initialization: built from YAML; no yolov8n.pt or other external pretrained checkpoint explicitly loaded
Training settings: epochs=300, batch=16, imgsz=640, optimizer=SGD, lr0=0.01, lrf=0.01, momentum=0.937, weight_decay=0.0005, warmup_epochs=3.0, patience=40, AMP=True, cache=False, workers=8
Data augmentation: hsv_h=0.015, hsv_s=0.7, hsv_v=0.4, translate=0.1, scale=0.5, fliplr=0.5, mosaic=1.0, mixup=0.0, copy_paste=0.0, close_mosaic=0
*Training command* :
model = YOLO('ultralytics/cfg/models/v8/yolov8n.yaml')
model.train(data='/split_4481/data.yaml', cache=False, imgsz=640, epochs=300, batch=16, close_mosaic=0, workers=8, optimizer='SGD', project='runs/train', name='exp', seed=<0|42|3407>)
*Evaluation settings*: best.pt from each training run, split=test, imgsz=640, batch=16
Evaluation command :
model = YOLO('runs/train/exp/weights/best.pt')
result = model.val(data='/split_4481/data.yaml', split='test', imgsz=640, batch=16, project='runs/val', name='exp')
FPS calculation: FPS = 1000 / (preprocess_time + inference_time + postprocess_time), where the three times are per-image times in milliseconds reported by Ultralytics.
**YOLOv8s**
Framework: Ultralytics
Package version: 8.2.50
Official release/tag: v8.2.50
Commit: 26a664f636e3de5c1afce0f55c9742e3b4d931ac
Model: YOLOv8s
Model configuration: ultralytics/cfg/models/v8/yolov8.yaml, scale=s
Training model entry: ultralytics/cfg/models/v8/yolov8s.yaml
Dataset: /split_4481/data.yaml
Seeds: 0, 42, 3407
Initialization: built from YAML; no yolov8s.pt or other external pretrained checkpoint explicitly loaded
Training settings: epochs=300, batch=16, imgsz=640, optimizer=SGD, lr0=0.01, lrf=0.01, momentum=0.937, weight_decay=0.0005, warmup_epochs=3.0, patience=40, AMP=True, cache=False, workers=8
Data augmentation: hsv_h=0.015, hsv_s=0.7, hsv_v=0.4, translate=0.1, scale=0.5, fliplr=0.5, mosaic=1.0, mixup=0.0, copy_paste=0.0, close_mosaic=0
*Training command* :
model = YOLO('ultralytics/cfg/models/v8/yolov8s.yaml')
model.train(data='/split_4481/data.yaml', cache=False, imgsz=640, epochs=300, batch=16, close_mosaic=0, workers=8, optimizer='SGD', project='runs/train', name='exp', seed=<0|42|3407>)
Evaluation settings: best.pt from each training run, split=test, imgsz=640, batch=16
*Evaluation command* :
model = YOLO('runs/train/exp/weights/best.pt')
result = model.val(data='/split_4481/data.yaml', split='test', imgsz=640, batch=16, project='runs/val', name='exp')
FPS calculation: FPS = 1000 / (preprocess_time + inference_time + postprocess_time), where the three times are per-image times in milliseconds reported by Ultralytics.
**YOLOv10n**
Framework: Ultralytics
Package version: 8.2.50
Official release/tag: v8.2.50
Commit: 26a664f636e3de5c1afce0f55c9742e3b4d931ac
Model: YOLOv10n
Model configuration: ultralytics/cfg/models/v10/yolov10n.yaml
Dataset: /split_4481/data.yaml
Seeds: 0, 42, 3407
Initialization: built from YAML; no yolov10n.pt or other external pretrained checkpoint explicitly loaded
Training settings: epochs=300, batch=16, imgsz=640, optimizer=SGD, lr0=0.01, lrf=0.01, momentum=0.937, weight_decay=0.0005, warmup_epochs=3.0, patience=40, AMP=True, cache=False, workers=8
Data augmentation: hsv_h=0.015, hsv_s=0.7, hsv_v=0.4, translate=0.1, scale=0.5, fliplr=0.5, mosaic=1.0, mixup=0.0, copy_paste=0.0, close_mosaic=0
*Training command*:
model = YOLO('ultralytics/cfg/models/v10/yolov10n.yaml')
model.train(data='/split_4481/data.yaml', cache=False, imgsz=640, epochs=300, batch=16, close_mosaic=0, workers=8, optimizer='SGD', project='runs/train', name='exp', seed=<0|42|3407>)
Evaluation settings: best.pt from each training run, split=test, imgsz=640, batch=16
*Evaluation command* :
model = YOLO('runs/train/exp/weights/best.pt')
result = model.val(data='/split_4481/data.yaml', split='test', imgsz=640, batch=16, project='runs/val', name='exp')
FPS calculation: FPS = 1000 / (preprocess_time + inference_time + postprocess_time), where the three times are per-image times in milliseconds reported by Ultralytics.
**YOLO11n**
Framework: Ultralytics
Package version: 8.3.9
Official release/tag: v8.3.9
Base commit: 50497218c25682458ea35b02dcc5d8a364f34591
Model: YOLO11n
Model configuration: ultralytics/cfg/models/11/yolo11.yaml, scale=n
Dataset: /split_4481/data.yaml
Seeds: 0, 42, 3407
Initialization: constructed from the YAML configuration; no yolo11n.pt or other external pretrained checkpoint was explicitly loaded.
**YOLO12n**
Framework base: Ultralytics 8.3.9
Base release/tag: v8.3.9
Ultralytics base release commit: 50497218c25682458ea35b02dcc5d8a364f34591
Model: YOLO12n
Model implementation: YOLOv12-Turbo architecture integrated into the Ultralytics 8.3.9 codebase
Original repository: sunsmarterjie/yolov12
Source revision: the exact upstream commit used for the local integration cannot be recovered because the archived local project does not retain Git history
Model configuration: ultralytics/cfg/models/12/yolo12.yaml
Dataset: /split_4481/data.yaml
Seeds: 0, 42, 3407
Initialization: constructed from the YAML configuration; no external pretrained checkpoint was explicitly loaded.
 **YOLO13n**
Framework base: Ultralytics 8.3.9
Base release/tag: v8.3.9
Ultralytics base release commit: 50497218c25682458ea35b02dcc5d8a364f34591
Model: YOLO13n
Model implementation: YOLOv13 architecture integrated into the Ultralytics 8.3.9 codebase
Original repository: iMoonLab/yolov13
Source revision: the exact upstream commit used for the local integration cannot be recovered because the archived local project does not retain Git history
Model configuration: ultralytics/cfg/models/13/yolo13.yaml
Local implementation modules: ultralytics/nn/extra_modules/yolov13.py
Dataset: /split_4481/data.yaml
Seeds: 0, 42, 3407
Initialization: constructed from the YAML configuration; no external pretrained checkpoint was explicitly loaded.
**Common training protocol for YOLO11n, YOLO12n, and YOLO13n**
Training settings: epochs=300, batch=16, imgsz=640, optimizer=SGD, lr0=0.01, lrf=0.01, momentum=0.937, weight_decay=0.0005, warmup_epochs=3.0, patience=40, AMP=True, cache=False, workers=4. Data augmentation: hsv_h=0.015, hsv_s=0.7, hsv_v=0.4, translate=0.1, scale=0.5, fliplr=0.5, mosaic=1.0, mixup=0.0, copy_paste=0.0, close_mosaic=0. 
*Training command * `from ultralytics import YOLO; model = YOLO('<model_config>'); model.train(data='/split_4481/data.yaml', cache=False, imgsz=640, epochs=300, batch=16, close_mosaic=0, workers=4, optimizer='SGD', patience=40, project='runs/train', name='exp', seed=<0|42|3407>)`. Model configurations: YOLO11n=`ultralytics/cfg/models/11/yolo11.yaml`; YOLO12n=`ultralytics/cfg/models/12/yolo12.yaml`; YOLO13n=`ultralytics/cfg/models/13/yolo13.yaml`.
**Common evaluation protocol for YOLO11n, YOLO12n, and YOLO13n**
The best checkpoint (`best.pt`) from each training run was evaluated on the test split of `/split_4481/data.yaml`. *Evaluation command :* `from ultralytics import YOLO; model = YOLO('<path_to_best.pt>'); result = model.val(data='/split_4481/data.yaml', split='test', imgsz=640, batch=16, project='runs/val', name='exp')`. Precision, Recall, F1-score, mAP50, and mAP50-95 were obtained from this test-set evaluation. Parameters and GFLOPs were obtained using `model_info(model.model)`. FPS was calculated as `FPS = 1000 / (preprocess_time + inference_time + postprocess_time)`, where preprocess_time, inference_time, and postprocess_time are the per-image times reported by Ultralytics in milliseconds.
**D-FINE-N**
Framework base: DEIM reproduction and modified implementation
Base repository: ShihuaHuang95/DEIM
Original D-FINE repository: Peterande/D-FINE
Source revision: the exact upstream commit/tag used for the archived local project cannot be recovered because the local directory does not retain Git history
Model: D-FINE-N
Model configuration: configs/dfine/dfine_hgnetv2_n_custom.yml
Dataset format: COCO
Dataset: /split_4481-coco/
Seeds: 0, 42, 3407
Initialization: HGNetv2 pretrained=False; no external pretrained backbone checkpoint was loaded
**D-FINE-S**
Framework base: DEIM reproduction and modified implementation
Base repository: ShihuaHuang95/DEIM
Original D-FINE repository: Peterande/D-FINE
Source revision: the exact upstream commit/tag used for the archived local project cannot be recovered because the local directory does not retain Git history
Model: D-FINE-S
Model configuration: configs/dfine/dfine_hgnetv2_s_custom.yml
Dataset format: COCO
Dataset: /split_4481-coco/
Seeds: 0, 42, 3407
Initialization: HGNetv2 pretrained=False; no external pretrained backbone checkpoint was loaded
**Common training protocol for D-FINE-N and D-FINE-S**
Common settings: epochs=300, total_batch_size=16, input_size=640×640, optimizer=AdamW, weight_decay=0.0001, warmup_duration=500 iterations, AMP=True, EMA=True, clip_max_norm=0.1, workers=4.
Model-specific learning rates: D-FINE-N uses lr=0.0008 and backbone_lr=0.0004; D-FINE-S uses lr=0.0002 and backbone_lr=0.0001.
Data augmentation: RandomPhotometricDistort(p=0.5), RandomZoomOut, RandomIoUCrop(p=0.8), RandomHorizontalFlip, Resize(640×640); RandomPhotometricDistort, RandomZoomOut, and RandomIoUCrop were disabled after epoch 288.
Training command:
CUDA_VISIBLE_DEVICES=0 torchrun --master_port=7777 --nproc_per_node=1 train.py -c <model_config> --use-amp --seed=<0|42|3407>
where <model_config> is:
D-FINE-N: configs/dfine/dfine_hgnetv2_n_custom.yml
D-FINE-S: configs/dfine/dfine_hgnetv2_s_custom.yml
**Common evaluation protocol for D-FINE-N and D-FINE-S**
Before evaluation, `val_dataloader` in `configs/dataset/custom_detection.yml` was changed from the validation split to the fixed test split. The best stage-1 checkpoint (`best_stg1.pth`) from each training run was then evaluated.
Evaluation command:
CUDA_VISIBLE_DEVICES=0 torchrun --master_port=7777 --nproc_per_node=1 train.py -c <model_config> --test-only -r <path_to_best_stg1.pth>
