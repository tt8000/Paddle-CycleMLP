import argparse
import paddle
import json
import time
import datetime
from pathlib import Path
import os

from engine import predict
from dataset import CycleMLPdataset, build_transfrom
from create import create_model


def get_args_parser():
    parser = argparse.ArgumentParser('CycleMLP Test script', add_help=False)
    parser.add_argument('--batch-size', default=64, type=int)

    # Model parameters
    parser.add_argument('--model', default='CycleMLP_B1', type=str, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument('--input-size', default=224, type=int, help='images input size')
    parser.add_argument('--drop', type=float, default=0.0, metavar='PCT',
                        help='Dropout rate (default: 0.)')
    parser.add_argument('--drop-path', type=float, default=0.1, metavar='PCT',
                        help='Drop path rate (default: 0.1)')
    parser.add_argument('--num-classes', type=int, default=1000,
                        help='number of categories')
    parser.add_argument('--model-pretrained', type=str, default='',
                        help='local model parameter path')

    # Augmentation parameters
    parser.add_argument('--train-interpolation', type=str, default='bicubic',
                        help='Training interpolation (random, bilinear, bicubic default: "bicubic")')

    # Dataset parameters
    parser.add_argument('--test-data-dir', default='./', type=str, help='image folder path')
    parser.add_argument('--test-txt-path', default='./test.txt', type=str,
                        help='image file name and label information file')
    parser.add_argument('--test-data-mode', default='test', type=str,
                        help="one of ['train', 'val', 'test'], the TXT file whether contains labels")
    parser.add_argument('--num_workers', default=0, type=int)
    
    parser.add_argument('--output_dir', default='./output',
                        help='path where to save, empty for no saving')

    return parser


def main(args):

    # 构建数据集
    test_transform = build_transfrom(is_train=False, args=args)
    test_dataset = CycleMLPdataset(args.test_data_dir, args.test_txt_path, mode=args.test_data_mode, transform=test_transform)
    data_loader_test = paddle.io.DataLoader(
        dataset=test_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        drop_last=False
    )
    
    # 构建模型
    print(f"Creating model: {args.model}")
    model = create_model(
        args.model,
        pretrained=args.model_pretrained,
        is_teacher=False,
        num_classes=args.num_classes,
        drop_rate=args.drop,
        drop_path_rate=args.drop_path)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    results_path = args.output_dir + '/results.txt'
    # 验证
    start_time = time.time()
    results = predict(data_loader_test, model) 
    with open(results_path, 'w') as f:
        for r in results:
            f.write(str(r) + '\n')

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Predict time {}'.format(total_time_str))


if __name__ == '__main__':
    parser = argparse.ArgumentParser('CycleMLP Test script', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)