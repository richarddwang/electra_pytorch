from functools import partial
from fastai.text.all import *
from hugdatafast.transform import CombineTransform

class MyConfig(dict):
  def __getattr__(self, name): return self[name]
  def __setattr__(self, name, value): self[name] = value

def adam_no_correction_step(p, lr, mom, step, sqr_mom, grad_avg, sqr_avg, eps, **kwargs):
    p.data.addcdiv_(grad_avg, (sqr_avg).sqrt() + eps, value = -lr)
    return p

def Adam_no_bias_correction(params, lr, mom=0.9, sqr_mom=0.99, eps=1e-5, wd=0.01, decouple_wd=True):
    "A `Optimizer` for Adam with `lr`, `mom`, `sqr_mom`, `eps` and `params`"
    cbs = [weight_decay] if decouple_wd else [l2_reg]
    cbs += [partial(average_grad, dampening=True), average_sqr_grad, step_stat, adam_no_correction_step]
    return Optimizer(params, cbs, lr=lr, mom=mom, sqr_mom=sqr_mom, eps=eps, wd=wd)

def linear_warmup_and_decay(pct, lr_max, total_steps, fake_total_steps=None, warmup_steps=None, warmup_pct=None, end_lr=0.0, decay_power=1):
  """ pct (float): fastai count it as ith_step/num_epoch*len(dl), so we can't just use pct when our num_epoch is fake.he ith_step is count from 0, """
  if warmup_pct: warmup_steps = int(warmup_pct * total_steps)
  step_i = round(pct * (fake_total_steps if fake_total_steps else total_steps)) + 1 # fastai count step from 0, so we have to add 1 back
  # According to the original source code, two schedules take effect at the same time, but decaying schedule will be neglible in the early time.
  decayed_lr = (lr_max-end_lr) * (1 - step_i/total_steps) ** decay_power + end_lr # https://www.tensorflow.org/api_docs/python/tf/compat/v1/train/polynomial_decay
  warmed_lr = decayed_lr * min(1.0, step_i/warmup_steps) # https://github.com/google-research/electra/blob/81f7e5fc98b0ad8bfd20b641aa8bc9e6ac00c8eb/model/optimization.py#L44
  return warmed_lr

def linear_warmup_and_then_decay(pct, lr_max, total_steps, fake_total_steps=None, warmup_steps=None, warmup_pct=None, end_lr=0.0, decay_power=1):
  """ pct (float): fastai count it as ith_step/num_epoch*len(dl), so we can't just use pct when our num_epoch is fake.he ith_step is count from 0, """
  if warmup_pct: warmup_steps = int(warmup_pct * total_steps)
  step_i = round(pct * (fake_total_steps if fake_total_steps else total_steps)) + 1 # fastai count step from 0, so we have to add 1 back
  if step_i <= warmup_steps: # warm up
    return lr_max * min(1.0, step_i/warmup_steps)
  else: # decay
    return (lr_max-end_lr) * (1 - (step_i-warmup_steps)/(total_steps-warmup_steps)) ** decay_power + end_lr

def load_part_model(file, model, prefix, device=None, strict=True):
  "assume `model` is part of (child attribute at any level) of model whose states save in `file`."
  distrib_barrier()
  if prefix[-1] != '.': prefix += '.'
  if isinstance(device, int): device = torch.device('cuda', device)
  elif device is None: device = 'cpu'
  state = torch.load(file, map_location=device)
  hasopt = set(state)=={'model', 'opt'}
  model_state = state['model'] if hasopt else state
  model_state = {k[len(prefix):] : v for k,v in model_state.items() if k.startswith(prefix)}
  get_model(model).load_state_dict(model_state, strict=strict)

def load_model_(learn, files, device=None, **kwargs):
  "if multiple file passed, then load and create an ensemble. Load normally otherwise"
  merge_out_fc = kwargs.pop('merge_out_fc', None)
  if not isinstance(files, list): 
    learn.load(files, device=device, **kwargs)
    return
  if device is None: device = learn.dls.device
  model = learn.model.cpu()
  models = [model, *(deepcopy(model) for _ in range(len(files)-1)) ]
  for f,m in zip(files, models):
    file = join_path_file(f, learn.path/learn.model_dir, ext='.pth')
    load_model(file, m, learn.opt, device='cpu', **kwargs)
  learn.model = Ensemble(models, device, merge_out_fc)
  return learn