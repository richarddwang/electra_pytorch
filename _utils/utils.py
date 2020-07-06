import time
from statistics import mean, stdev
import torch
# from fastai2.callback.all import * # name 'store_attr' is not defined
from fastai2.text.all import * 

class RunSteps(Callback):
  toward_end = True
  
  def __init__(self, n_steps, save_points=None, base_name=None, no_val=True):
    """
    Args:
      `n_steps` (`Int`): Run how many steps, could be larger or smaller than `len(dls.train)`
      `savepoints` 
      - (`List[Float]`): save when reach one of percent specified.
      - (`List[Int]`): save when reache one of steps specified
      `base_name` (`String`): a format string with `{percent}` to be passed to `learn.save`.
    """
    if save_points is None: save_points = []
    else:
      assert '{percent}' in base_name
      save_points = [ s if isinstance(s,int) else int(n_steps*s) for s in save_points ]
      assert max(save_points) <= n_steps
    store_attr(self, 'n_steps,save_points,base_name,no_val')

  def after_batch(self):
    if self.train_iter in self.save_points:
      percent = round((self.train_iter/self.n_steps)*100)
      self.learn.save(self.base_name.format(percent=f'{percent}%'))
    if self.train_iter == self.n_steps:
      raise CancelFitException

  def after_train(self):
    if self.no_val:
      if self.train_iter == self.n_steps:
        pass # CancelFit is raised, don't overlap it with CancelEpoch
      else:
        raise CancelEpochException

_MESSAGE = [
  'dl.train load a batch + begin_batch',
  'forward + after_pred',
  'loss calculation + after_loss',
  'backward + after_backward',
  'parameter updating + after_step',
  'after_batch',
]

@delegates()
class Timer(RunSteps):
  toward_end=True

  def __init__(self, n_steps, ignore_first_n=1, break_after=None, precision=3, **kwargs):
    """
    Args:
      `n_steps`: Average on how many training steps.
      `ignore_first_n`: Not use first n steps to average. Setting it at least 1 to avoid counting initilization time of dataloader is suggested.
      `break_after`: one of ['begin_batch',...'after_batch']
      `precision`
    """
    steps = ignore_first_n + n_steps
    super().__init__(steps, **kwargs)
    store_attr(self, 'steps,break_after,ignore_first_n,precision')

  def time_delta(self): 
    delta = time.time() - self.timepoint
    self.timepoint = time.time()
    return delta

  def begin_fit(self):
    self.times = [ [] for _ in range(6)]
    self.timepoint = time.time()
  def begin_batch(self):
    self.times[0].append(self.time_delta())
    if self.break_after=='begin_batch': raise CancelBatchException
  def after_pred(self):
    self.times[1].append(self.time_delta())
    if self.break_after=='after_pred': raise CancelBatchException
  def after_loss(self): 
    self.times[2].append(self.time_delta())
    if self.break_after=='after_loss': raise CancelBatchException
  def after_backward(self): 
    self.times[3].append(self.time_delta())
    if self.break_after=='after_backward': raise CancelBatchException
  def after_step(self):
    self.times[4].append(self.time_delta())
    if self.break_after=='after_step': raise CancelBatchException
  def after_batch(self):
    if self.break_after=='after_batch' or not self.break_after:
      self.times[5].append(self.time_delta())
    if self.train_iter == self.steps:
      self.show()
    super().after_batch()

  def show(self):
    print(f"show average and standard deviation of step {self.ignore_first_n+1} ~ step {self.train_iter} (total {self.steps-self.ignore_first_n} training steps)")
    # print for each stage
    for i, deltas in enumerate(self.times):
      if len(deltas)==0: time_message = "Skipped or Exception raised by callbacks ran before Timer."
      else:
        m,s = mean(deltas[self.ignore_first_n:]), stdev(deltas[self.ignore_first_n:])
        time_message = f"avg {round(m, self.precision)} secs ± stdev {round(s, self.precision)}"
      print(f"{(_MESSAGE[i]+':'):36} {time_message}")
    # print for total
    times = list(filter(None, self.times))
    ## Some callback (e.g. MixedPrecisionCallback) might skip some stage "sometimes", so the length might be not equal 
    max_len = max( len(deltas) for deltas in times )
    for i, deltas in enumerate(times):
      if len(deltas) < max_len: times[i] += [0]*(max_len - len(deltas))
    ## calculate
    times = torch.tensor(times)[:,self.ignore_first_n:]
    total_m, total_s = times.sum(0).mean().item(), times.sum(0).std().item()
    print(f"Total: avg {round(total_m, self.precision)} secs ± stdev {round(total_s, self.precision)} secs")
