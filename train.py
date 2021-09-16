import argparse
import datetime
import os
import time
import paddle
import paddle.distributed as dist
import json
from pathlib import Path

from engine import train_one_epoch, evaluate
import utils
from dataset import CycleMLPdataset, build_transfrom
from losses import DistillationLoss, SoftTargetCrossEntropy, LabelSmoothingCrossEntropy
from data import Mixup
from create import create_model, create_optimizer_scheduler


def get_args_parser():
    parser = argparse.ArgumentParser('CycleMLP training and evaluation script', add_help=False)
    parser.add_argument('--batch-size', default=64, type=int)
    parser.add_argument('--epochs', default=300, type=int)

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

    # Optimizer parameters
    parser.add_argument('--opt', default='AdamW', type=str, metavar='OPTIMIZER',
                        help='Optimizer (default: "AdamW"')
    parser.add_argument('--opt-eps', default=1e-8, type=float, metavar='EPSILON',
                        help='Optimizer Epsilon (default: 1e-8)')
    parser.add_argument('--opt-beta1', default=None, type=float, nargs='+', metavar='BETA1',
                        help='Optimizer Beta1 (default: None, use opt default)')
    parser.add_argument('--opt-beta2', default=None, type=float, nargs='+', metavar='BETA2',
                    help='Optimizer Beta1 (default: None, use opt default)')
    parser.add_argument('--clip-grad', type=float, default=None, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--weight-decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')

    # Learning rate schedule parameters
    parser.add_argument('--sched', default='CosineAnnealingDecay', type=str, metavar='SCHEDULER',
                        help='LR scheduler (default: "CosineAnnealingDecay"')
    parser.add_argument('--lr', type=float, default=5e-4, metavar='LR',
                        help='learning rate (default: 5e-4)')
    parser.add_argument('--t-max', default=300, type=int,
                        help='the upper limit for training is half the cosine decay period, the default equal epochs')
    parser.add_argument('--eta-min', default=0, type=float,
                        help='the minimum value of the learning rate is ηmin in the formula, the default value is 0')
    parser.add_argument('--last-epoch', default=-1, type=int,
                        help='the epoch of the previous round is set to the epoch of the previous round when training is restarted.\
                        the default value is -1, indicating the initial learning rate ')

    # Augmentation parameters
    parser.add_argument('--color-jitter', type=float, default=0.4, metavar='PCT',
                        help='Color jitter factor (default: 0.4)')
    parser.add_argument('--aa', type=str, default='rand-m9-mstd0.5-inc1', metavar='NAME',
                        help='Use AutoAugment policy. "v0" or "original". " + \
                             "(default: rand-m9-mstd0.5-inc1)'),
    parser.add_argument('--smoothing', type=float, default=0.1, help='Label smoothing (default: 0.1)')
    parser.add_argument('--train-interpolation', type=str, default='bicubic',
                        help='Training interpolation (random, bilinear, bicubic default: "bicubic")')

    # * Random Erase params
    parser.add_argument('--reprob', type=float, default=0.25, metavar='PCT',
                        help='Random erase prob (default: 0.25)')

    # * Mixup params
    parser.add_argument('--mixup', type=float, default=0.8,
                        help='mixup alpha, mixup enabled if > 0. (default: 0.8)')
    parser.add_argument('--cutmix', type=float, default=1.0,
                        help='cutmix alpha, cutmix enabled if > 0. (default: 1.0)')
    parser.add_argument('--cutmix-minmax', type=float, nargs='+', default=None,
                        help='cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)')
    parser.add_argument('--mixup-prob', type=float, default=1.0,
                        help='Probability of performing mixup or cutmix when either/both is enabled')
    parser.add_argument('--mixup-switch-prob', type=float, default=0.5,
                        help='Probability of switching to cutmix when both mixup and cutmix enabled')
    parser.add_argument('--mixup-mode', type=str, default='batch',
                        help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"')

    # Distillation parameters
    parser.add_argument('--teacher-model', default='RegNetX_4GF', type=str, metavar='MODEL',
                        help='Name of teacher model to train (default: "RegNetX_4GF"')
    parser.add_argument('--teacher-pretrained', default=None, type=str,
                        help='teacher model parameters must be downloaded locally')
    parser.add_argument('--distillation-type', default='none', choices=['none', 'soft', 'hard'], type=str, help="")
    parser.add_argument('--distillation-alpha', default=0.5, type=float, help="")
    parser.add_argument('--distillation-tau', default=1.0, type=float, help="")

    # * Finetuning params
    parser.add_argument('--finetune', default='', help='finetune from checkpoint')

    # Dataset parameters
    parser.add_argument('--train-data-dir', default='./', type=str, help='image folder path')
    parser.add_argument('--train-txt-path', default='./train.txt', type=str,
                        help='image file name and label information file')
    parser.add_argument('--train-data-mode', default='train', type=str,
                        help="one of ['train', 'val', 'test'], the TXT file whether contains labels")

    parser.add_argument('--val-data-dir', default='./', type=str, help='image folder path')
    parser.add_argument('--val-txt-path', default='./val.txt', type=str,
                        help='image file name and label information file')
    parser.add_argument('--val-data-mode', default='val', type=str,
                        help="one of ['train', 'val', 'test'], the TXT file whether contains labels")
    parser.add_argument('--num_workers', default=0, type=int)
    parser.add_argument('--output_dir', default='./output',
                        help='path where to save, empty for no saving')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')

    # distributed training
    parser.add_argument('--is_distributed', default=False, type=bool,
                        help='whether to enable single-machine multi-card training')

    # custom parameters
    parser.add_argument('--is_amp', default=False, type=bool,
                        help='whether to enable automatic mixing precision training')
    parser.add_argument('--init_loss_scaling', default=1024, type=float,
                        help='initial Loss Scaling factor. The default value is 1024')
    return parser


def main(args):

    print(args)
    
    if args.distillation_type != 'none' and args.finetune:
        raise NotImplementedError("Finetuning with distillation not yet supported")
    

    # 构建数据
    train_transform = build_transfrom(is_train=True,args=args)
    train_dataset = CycleMLPdataset(args.train_data_dir, args.train_txt_path, mode=args.train_data_mode, transform=train_transform)
    
    data_loader_train = paddle.io.DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        drop_last=True,
    )

    val_transform = build_transfrom(is_train=False, args=args)
    val_dataset = CycleMLPdataset(args.val_data_dir, args.val_txt_path, mode=args.val_data_mode, transform=val_transform)
    data_loader_val = paddle.io.DataLoader(
        dataset=val_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        drop_last=False
    )

    # mixup混类数据增强
    mixup_fn = None
    mixup_active = args.mixup > 0 or args.cutmix > 0. or args.cutmix_minmax is not None
    if mixup_active:
        mixup_fn = Mixup(
            mixup_alpha=args.mixup, cutmix_alpha=args.cutmix, cutmix_minmax=args.cutmix_minmax,
            prob=args.mixup_prob, switch_prob=args.mixup_switch_prob, mode=args.mixup_mode,
            label_smoothing=args.smoothing, num_classes=args.num_classes)

    print(f"Creating model: {args.model}")
    model = create_model(
        args.model,
        pretrained=args.model_pretrained,
        is_teacher=False,
        num_classes=args.num_classes,
        drop_rate=args.drop,
        drop_path_rate=args.drop_path)

    # 配置蒸馏模型
    teacher_model = None
    if args.distillation_type != 'none':
        print(f"Creating teacher model: {args.teacher_model}")
        teacher_model = create_model(
            args.teacher_model,
            pretrained=False,
            is_teacher=True,
            class_num=args.num_classes
        )
        if os.path.exists(args.teacher_pretrained):
            teacher_model.set_state_dict(paddle.load(args.teacher_pretrained))
        teacher_model.eval() 

    get_world_size = 1
    # 是否分布式
    if args.is_distributed:
        dist.init_parallel_env()
        model = paddle.DataParallel(model)
        teacher_model = paddle.DataParallel(teacher_model)
        get_world_size = dist.get_world_size()

    # finetune 微调
    if args.finetune:
        if os.path.exists(args.finetune):
            print('You must download the finetune model and place it locally.')
        else:
            checkpoint = paddle.load(args.finetune)

        checkpoint_model = checkpoint
        state_dict = model.state_dict()
        for k in ['head.weight', 'head.bias', 'head_dist.weight', 'head_dist.bias']:
            if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
                print(f"Removing key {k} from pretrained checkpoint")
                del checkpoint_model[k]
    
        # interpolate position embedding
        pos_embed_checkpoint = checkpoint_model['pos_embed']
        embedding_size = pos_embed_checkpoint.shape[-1]
        num_patches = model.patch_embed.num_patches
        num_extra_tokens = model.pos_embed.shape[-2] - num_patches
        # height (== width) for the checkpoint position embedding
        orig_size = int((pos_embed_checkpoint.shape[-2] - num_extra_tokens) ** 0.5)
        # height (== width) for the new position embedding
        new_size = int(num_patches ** 0.5)
        # class_token and dist_token are kept unchanged
        extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
        # only the position tokens are interpolated
        pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
        pos_tokens = pos_tokens.reshape((-1, orig_size, orig_size, embedding_size)).transpose((0, 3, 1, 2))
        pos_tokens = paddle.nn.functional.interpolate(
            pos_tokens, size=(new_size, new_size), mode='bicubic', align_corners=False)
        pos_tokens = pos_tokens.transpose((0, 2, 3, 1)).flatten(1, 2)
        new_pos_embed = paddle.concat((extra_tokens, pos_tokens), axis=1)
        checkpoint_model['pos_embed'] = new_pos_embed

        model.set_state_dict(checkpoint_model)

    # 优化器配置
    linear_scaled_lr = args.lr * args.batch_size * get_world_size / 512.0
    args.lr = linear_scaled_lr
    optimizer, scheduler = create_optimizer_scheduler(args, model)    

    # setup automatic mixed-precision (AMP) loss scaling and op casting
    loss_scaler = None
    if args.is_amp:
        loss_scaler = paddle.amp.GradScaler(init_loss_scaling=args.init_loss_scaling)


    n_parameters = sum(p.numel() for p in model.parameters() if not p.stop_gradient).numpy()[0]
    print('number of params:', n_parameters)
    print('=' * 30)

    if args.mixup > 0.:
        # smoothing is handled with mixup label transform
        criterion = SoftTargetCrossEntropy()
    elif args.smoothing:
        criterion = LabelSmoothingCrossEntropy(smoothing=args.smoothing)
    else:
        criterion = paddle.nn.CrossEntropyLoss()

    # wrap the criterion in our custom DistillationLoss, which
    # just dispatches to the original criterion if args.distillation_type is 'none'
    criterion = DistillationLoss(
        criterion, teacher_model, args.distillation_type, args.distillation_alpha, args.distillation_tau
    ) 

    # 训练
    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    max_accuracy = 0.0
    log_path = args.output_dir + "/train_log.txt"
    for epoch in range(args.start_epoch, args.epochs):

        train_stats = train_one_epoch(
            model, criterion, data_loader_train,
            optimizer, epoch, log_path, scheduler, 
            loss_scaler, mixup_fn, args.is_distributed)

        # 参数保存
        if args.output_dir:
            utils.save_on_master({
                'pdparams': model.state_dict(),
                'pdopt': optimizer.state_dict(),
                'pdsched': scheduler.state_dict(),
                'pdepoch': epoch,
                'pdscaler': loss_scaler.state_dict() if loss_scaler is not None else None,
                'pdargs': args,
            }, args.output_dir + f'/checkpoint_{epoch}')
        # 验证
        test_stats = evaluate(data_loader_val, model, log_path)
        print(f"Accuracy of the network on the {len(val_dataset)} test images: {test_stats['acc1']:.1f}%")
        max_accuracy = max(max_accuracy, test_stats["acc1"])
        print(f'Max accuracy: {max_accuracy:.2f}%')

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     **{f'test_{k}': v for k, v in test_stats.items()},
                     'epoch': epoch,
                     'n_parameters': str(n_parameters)}

        for key in log_stats:
            print(type(log_stats[key]))
        if args.output_dir and utils.is_main_process():
            with open(log_path, 'a') as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    parser = argparse.ArgumentParser('CycleMLP training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)