"""
Train, eval and test functions used in main.py
"""
import math
import sys
import paddle
from paddle.metric import accuracy
from tqdm import tqdm

import utils


def train_one_epoch(model, criterion, data_loader, optimizer, epoch, log_path,
                    scheduler=None, loss_scaler=None, mixup_fn=None, is_distributed=False):

    model.train()
    metric_logger = utils.MetricLogger(log_path, delimiter="  ", is_distributed=is_distributed)
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)

        # 是否开启混合精度训练
        if loss_scaler:
            with paddle.amp.auto_cast():
                outputs = model(samples)
                loss = criterion(samples, outputs, targets)

            scaled = loss_scaler.scale(loss)
            scaled.backward()

            loss_scaler.minimize(optimizer, scaled)
        else:
            outputs = model(samples)
            loss = criterion(samples, outputs, targets)
            loss.backward()
            optimizer.step()

        loss_value = loss.numpy()[0]
        optimizer.clear_grad()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)
        # 更新日志
        metric_logger.update(loss=loss_value)
        metric_logger.update(lr=optimizer.get_lr())
    if scheduler:
        scheduler.step()
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@paddle.no_grad()
def evaluate(data_loader, model, log_path):
    criterion = paddle.nn.CrossEntropyLoss()

    metric_logger = utils.MetricLogger(log_path, delimiter="  ")
    header = 'Val:'

    # switch to evaluation mode
    model.eval()

    for images, target in metric_logger.log_every(data_loader, 10, header):
        output = model(images)
        loss = criterion(output, target)

        target = target.reshape((-1, 1))
        acc1 = accuracy(output, target, k=1)
        acc5 = accuracy(output, target, k=5)

        batch_size = images.shape[0]
        metric_logger.update(loss=loss.numpy()[0])
        metric_logger.meters['acc1'].update(acc1.numpy()[0], n=batch_size)
        metric_logger.meters['acc5'].update(acc5.numpy()[0], n=batch_size)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
          .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@paddle.no_grad()
def predict(data_loader, model):

    header = 'Test:'
    # switch to evaluation mode
    model.eval()
    results = []
    for images in tqdm(data_loader):
        output = model(images)
        results.extend(list(paddle.argmax(output).reshape((-1).numpy())))

    return results