import argparse
import warnings
warnings.filterwarnings('ignore')
from ultralytics import RTDETR

# onnx onnxsim onnxruntime onnxruntime-gpu

# 导出参数官方详解链接：https://docs.ultralytics.com/modes/export/#usage-examples


def parse_args():
    parser = argparse.ArgumentParser(description='Export an RT-DETR-GMI checkpoint.')
    parser.add_argument('--model', required=True, help='Checkpoint path to export.')
    parser.add_argument('--format', default='onnx', help='Export format, for example onnx, engine, torchscript.')
    parser.add_argument('--imgsz', type=int, default=640, help='Export image size.')
    parser.add_argument('--device', default=None, help="CUDA device, for example '0'.")
    parser.add_argument('--simplify', action=argparse.BooleanOptionalAction, default=True, help='Simplify ONNX graph when supported.')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    model = RTDETR(args.model)
    export_kwargs = {'format': args.format, 'imgsz': args.imgsz, 'simplify': args.simplify}
    if args.device:
        export_kwargs['device'] = args.device
    model.export(**export_kwargs)
