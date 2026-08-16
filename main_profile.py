import argparse
import warnings
warnings.filterwarnings('ignore')
from ultralytics import RTDETR


def parse_args():
    parser = argparse.ArgumentParser(description='Profile an RT-DETR-GMI model definition or checkpoint.')
    parser.add_argument('--model', default='RT-DETR-GMI.yaml', help='Model YAML or checkpoint path.')
    parser.add_argument('--imgsz', type=int, nargs=2, default=[640, 640], metavar=('HEIGHT', 'WIDTH'), help='Profile image size.')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    model = RTDETR(args.model)
    model.model.eval()
    model.info(detailed=True)
    try:
        model.profile(imgsz=args.imgsz)
    except Exception as e:
        print(e)
        pass
    print('after fuse:', end='')
    model.fuse()
