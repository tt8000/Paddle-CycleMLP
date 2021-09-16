import io
import os
import time
from collections import defaultdict, deque
import datetime
import paddle
import paddle.distributed as dist
from collections import Counter
import typing
from typing import Any, List

IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)


def is_main_process():
    return dist.get_rank() == 0


class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self, is_distributed):
        """
        Warning: does not synchronize the deque!
        """
        if not is_distributed:
            return
        t = paddle.to_tensor([self.count, self.total], dtype='float64')
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self):
        d = paddle.to_tensor(list(self.deque))
        return d.median().numpy()[0]

    @property
    def avg(self):
        d = paddle.to_tensor(list(self.deque), dtype='float32')
        return d.mean().numpy()[0]

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value)


class MetricLogger(object):
    def __init__(self, log_path, delimiter="\t", is_distributed=False):
        self.meters = defaultdict(SmoothedValue)
        self.log_path = log_path
        self.delimiter = delimiter
        self.is_distributed = is_distributed


    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, paddle.Tensor):
                v = v.numpy()
            if 'float' in str(type(v)):
                v = float(v)
            else:
                v = int(v)
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, attr))

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(
                "{}: {}".format(name, str(meter))
            )
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes(self.is_distributed)

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None):
        i = 0
        if not header:
            header = ''
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt='{avg:.4f}')
        data_time = SmoothedValue(fmt='{avg:.4f}')
        space_fmt = ':' + str(len(str(len(iterable)))) + 'd'
        log_msg = [
            header,
            '[{0' + space_fmt + '}/{1}]',
            'eta: {eta}',
            '{meters}',
            'time: {time}',
            'data-time: {data}'
        ]
        log_msg = self.delimiter.join(log_msg)
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0 or i == len(iterable) - 1:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                with open(self.log_path, 'a') as f:
                    f.write(log_msg.format(
                            i, len(iterable), eta=eta_string,
                            meters=str(self),
                            time=str(iter_time), data=str(data_time)) + '\n')
                print(log_msg.format(
                    i, len(iterable), eta=eta_string,
                    meters=str(self),
                    time=str(iter_time), data=str(data_time)))
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('{} Total time: {} ({:.4f} s / it)'.format(
            header, total_time_str, total_time / len(iterable)))


def save_on_master(d_params, s_dir):
    if not os.path.exists(s_dir):
        os.makedirs(s_dir)
    for key in d_params:
        paddle.save(d_params[key], s_dir + f'/{key[2:]}.{key}')