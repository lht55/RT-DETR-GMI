import argparse
import warnings

warnings.filterwarnings('ignore')
from ultralytics import RTDETR


def parse_args():
    parser = argparse.ArgumentParser(description='Train RT-DETR-GMI.')
    parser.add_argument('--model', default='RT-DETR-GMI.yaml', help='Model YAML or checkpoint path.')
    parser.add_argument('--data', required=True, help='Dataset YAML path, for example datasets/my_data/data.yaml.')
    parser.add_argument('--pretrained', default=None, help='Optional pretrained checkpoint to load before training.')
    parser.add_argument('--imgsz', type=int, default=640, help='Training image size.')
    parser.add_argument('--epochs', type=int, default=300, help='Number of training epochs.')
    parser.add_argument('--batch', type=int, default=16, help='Batch size.')
    parser.add_argument('--workers', type=int, default=4, help='Number of dataloader workers.')
    parser.add_argument('--device', default=None, help="CUDA device, for example '0' or '0,1'.")
    parser.add_argument('--project', default='runs/train', help='Output project directory.')
    parser.add_argument('--name', default='exp', help='Run name.')
    parser.add_argument('--resume', default=None, help='Optional checkpoint path for resuming training.')
    parser.add_argument('--patience', type=int, default=None, help='Early stopping patience.')
    parser.add_argument('--cache', action='store_true', help='Cache dataset images for training.')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    model = RTDETR(args.model)
    if args.pretrained:
        model.load(args.pretrained)

    train_kwargs = {
        'data': args.data,
        'cache': args.cache,
        'imgsz': args.imgsz,
        'epochs': args.epochs,
        'batch': args.batch,
        'workers': args.workers,
        'project': args.project,
        'name': args.name,
    }
    if args.device:
        train_kwargs['device'] = args.device
    if args.resume:
        train_kwargs['resume'] = args.resume
    if args.patience is not None:
        train_kwargs['patience'] = args.patience

    model.train(**train_kwargs)
