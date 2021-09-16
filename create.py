import paddle
from cycle_mlp import *
from ppcls.arch.backbone.model_zoo.regnet import *


def create_model(model_name,
        pretrained=False,
        is_teacher=False,
        **kwargs):
    
    if is_teacher:
        model = eval(model_name)(pretrained=pretrained, use_ssld=False, **kwargs)
    else:
        model = eval(model_name)(**kwargs)
    
    if pretrained and os.path.exists(pretrained):
        model.set_state_dict(paddle.load(pretrained))
    return model


def optimizer_kwargs(cfg):
    """ cfg/argparse to kwargs helper
    Convert optimizer args in argparse args or cfg like object to keyword args for updated create fn.
    """
    kwargs = dict()
    kwargs['beta1'] = cfg.opt_beta1 if cfg.opt_beta1 != None else 0.9
    kwargs['beta2'] = cfg.opt_beta2 if cfg.opt_beta2 != None else 0.999
    kwargs['epsilon'] = cfg.opt_eps
    kwargs['weight_decay'] = cfg.weight_decay
    return kwargs


def scheduler_kwargs(cfg):
    kwargs = dict()
    kwargs['learning_rate'] = cfg.lr
    kwargs['T_max'] = cfg.t_max
    kwargs['eta_min'] = cfg.eta_min
    kwargs['last_epoch'] = cfg.last_epoch
    return kwargs


def create_optimizer_scheduler(cfg, model):
    opt = cfg.opt
    sched = cfg.sched
    assert opt == 'AdamW', 'Currently, only AdamW is supported !'
    assert sched == 'CosineAnnealingDecay', 'Currently, only CosineAnnealingDecay is supported !'
    clip_grad = cfg.clip_grad
    if clip_grad != None:
        clip_grad = paddle.nn.ClipGradByNorm(clip_grad)
    opt_kwargs = optimizer_kwargs(cfg)
    sched_kwargs = scheduler_kwargs(cfg)
    scheduler = paddle.optimizer.lr.CosineAnnealingDecay(**sched_kwargs)
    optimizer = paddle.optimizer.AdamW(learning_rate=scheduler, parameters=model.parameters(), grad_clip=clip_grad, **opt_kwargs)
    return optimizer, scheduler