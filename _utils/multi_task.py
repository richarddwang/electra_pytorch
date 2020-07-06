from fastprogress.fastprogress import format_time
from itertools import cycle
import time
from fastai2.text.all import *

"""
* (Original) all metrics include loss accumulate when `after_batch` -> (New) Because `MultiTaskLearner` will loop for tasks and `self.pred` will be overwrited within the loop, so **metrics** catch it when `after_pred`, before entering the next loop. But **loss** still accumulate when `after_batch` to catch merged loss.

* Helper function `header` to infer values for `self.metric_names`. Extract this work out from `begin_fit`, so `MultiTaskRecorde` can just overwrite it to create multi task metrics header. 

* `MyProgressCallback` is just to push progress cb after `MyRecorder`

* Unlike original `Recorder`, *metrics* here is refer to all metrics **but not `loss`**. Specially keep this in mind when looking at `MultiTaskRecorder`'s code
"""

def _maybe_item(t):
    t = t.value
    return t.item() if isinstance(t, Tensor) and t.numel()==1 else t

class MyRecorder(Recorder):
  remove_on_fetch,run_after = True,TrainEvalCallback
  def __init__(self, show_train_loss=True, train_do_metric=False, show_valid_loss=True, valid_do_metric=True, show_time=True, beta=0.98):
    assert show_train_loss or train_do_metric or show_valid_loss or valid_do_metric or show_time
    store_attr(self, 'show_train_loss,train_do_metric,show_valid_loss,valid_do_metric,show_time,beta')

  def begin_fit(self):
    # init
    self.train_loss = AvgSmoothLoss(beta=self.beta)
    self.valid_loss = AvgLoss()
    self.lrs,self.iters,self.losses,self.values = [],[],[],[]
    self.train_loss.reset()
    # header
    self.metric_names = self.header()

  def header(self):
    header = L(['epoch'])
    if self.show_train_loss: header += 'train_loss'
    if self.train_do_metric: header += self.metrics.attrgot('name')
    if self.show_valid_loss: header += 'valid_loss'
    if self.valid_do_metric: header += self.metrics.attrgot('name')
    if self.show_time: header += 'time'
    return header

  def begin_epoch(self):
    "Set timer if `self.add_time=True`"
    self.cancel_train,self.cancel_valid = False,False
    self.epoch_start_time = time.time()
    self.log = L(getattr(self, 'epoch', 0))
  
  def after_pred(self):
    if len(self.yb) == 0: return
    if (self.training and self.train_do_metric) or (not self.training and self.valid_do_metric):
      for met in self._metrics: met.accumulate(self.learn)

  def after_batch(self):
    if len(self.yb) == 0: return
    if self.training:
      self.train_loss.accumulate(self.learn)
      self.lrs.append(self.opt.hypers[-1]['lr'])
      self.losses.append(self.train_loss.value)
      self.learn.smooth_loss = self.train_loss.value
    else:
      self.valid_loss.accumulate(self.learn)      

  def begin_train   (self):
    if self.train_do_metric: self._metrics.map(Self.reset())
  def begin_validate(self):
    if self.valid_do_metric: self._metrics.map(Self.reset()); self.valid_loss.reset()
  def after_train   (self):
    if self.show_train_loss: self.log += _maybe_item(self.train_loss)
    if self.train_do_metric: self.log += self._metrics.map(_maybe_item)
  def after_validate(self):
    if self.show_valid_loss: self.log += _maybe_item(self.valid_loss)
    if self.valid_do_metric: self.log += self._metrics.map(_maybe_item)

  def after_epoch(self):
    "Store and log the loss/metric values"
    self.learn.final_record = self.log[1:].copy() # first element is epoch
    self.values.append(self.learn.final_record)
    if self.show_time: self.log.append(format_time(time.time() - self.epoch_start_time))
    self.logger(self.log)
    self.iters.append(self.train_loss.count)
    
  @property
  def _metrics(self):
    if self.training and getattr(self, 'cancel_train', False): return L()
    if not self.training and getattr(self, 'cancel_valid', False): return L()
    return self.metrics

  @property
  def name(self): return 'recorder'

class MyProgressCallback(ProgressCallback):
  run_after=MyRecorder

"""
* Extract loss calculating out from `one_batch`, because `MultiTaskLearner` will merge different tasks losses to calculate loss, but after it is just the same as single task.

* Get the batch by `next` the iterator of `dl` in `one_loss` but not directly loop dl in `all_batches`. So `MultiTaksLearner` can overwrite the iteraotr of dl and calculate the batches of the tasks one by one. 
"""
  
class MyLearner(Learner):
  def __init__(self,*args, **kwargs):
    super().__init__(*args,**kwargs)
    self.remove_cbs([Recorder,ProgressCallback])
    self.add_cbs([MyRecorder(),MyProgressCallback()])

  def all_batches(self):
    self.n_iter = len(self.dl)
    self.dl_iter = iter(self.dl)
    for self.iter in range(self.n_iter): self.one_batch()

  def one_loss(self):
    b = next(self.dl_iter)
    self._split(b);                                  self('begin_batch')
    self.pred = self.model(*self.xb);                self('after_pred')
    if len(self.yb) == 0: return
    self.loss = self.loss_func(self.pred, *self.yb); self('after_loss')
    
  def one_batch(self):
    try:
      self.one_loss()
      if not self.training: return
      self.loss.backward();                            self('after_backward')
      self.opt.step();                                 self('after_step')
      self.opt.zero_grad()
    except CancelBatchException:                         self('after_cancel_batch')
    finally:                                             self('after_batch')

"""
* Overwrite the `header` function to use display layout for multi task setting.

* We can record only one training loss but we can record every task's valid loss.

* When refer to `valid_loss`, it means get that task's `valid_loss`(implemented by `@property def valid_loss`), which is not the case of `training loss`.

* Here rewrite the `metrics` property, so we can get only the current task's metrics to record.

* In `MultiTaskLearner` you can specify `tasks_dont_measure` so thos tasks won't measure metrics in training, and won't do validate for thos tasks, and surely their metrics and valid loss are not record.

* You can also specify `tasks_dont_metric`, those tasks won't have metric logged, but different with `tasks_dont_measure` will **still show `train_loss` and `valid_loss`**.
"""
    
class MulitTaskRecorder(MyRecorder):
  @delegates(MyRecorder.__init__)
  def __init__(self, tasks_dont_metric=[], **kwargs):
    super().__init__(**kwargs)
    store_attr(self, 'tasks_dont_metric')

  def begin_fit(self):
    # init
    self.train_loss = AvgSmoothLoss(beta=self.beta)
    self.valid_losses = L([AvgLoss() for _ in range(len(self.learn))])
    self.lrs,self.iters,self.losses,self.values = [],[],[],[]
    self.train_loss.reset()
    # header
    self.metric_names = self.header()
  
  @property
  def valid_loss(self): return self.valid_losses[self.current_task_idx]

  def dont_measure(self, i): return i in self.tasks_dont_measure or self.task_names[i] in self.tasks_dont_measure
  def dont_metric(self, i): return i in self.tasks_dont_metric or self.task_names[i] in self.tasks_dont_metric
  
  def header(self):
    header = L(['epoch'])
    if self.show_train_loss: header += 'train_loss'
    if self.train_do_metric:
      for i, metrics in enumerate(self.multi_metrics):
        if self.dont_measure(i) or self.dont_metric(i): continue
        for metric in metrics:
          header += f'{self.task_names[i]}:{metric.name}'
    if self.show_valid_loss or self.valid_do_metric:
      for i, metrics in enumerate(self.multi_metrics):
        if self.dont_measure(i): continue
        if self.show_valid_loss: header += f'{self.task_names[i]}:valid_loss'
        if self.dont_metric(i): continue
        if self.valid_do_metric:
          for metric in metrics:
            header += f'{self.task_names[i]}:{metric.name}'
    if self.show_time: header += 'time'
    return header

  @property
  def _metrics(self):
    "return metrics of current task which is not in `not_do_task`"
    if self.training and getattr(self, 'cancel_train', False): return L()
    if not self.training and getattr(self, 'cancel_valid', False): return L()
    mets = L()
    for i, metrics in enumerate(self.multi_metrics):
      if self.dont_measure(i) or self.dont_metric(i): continue
      if self.current_task_idx==i or self.current_task_idx is None:
        mets += metrics
    return mets

class MyMyProgressCallback(ProgressCallback):
  run_after=MulitTaskRecorder

def not_cycle_infinite(dl):
  for b in dl: yield b
  while True: yield None

"""
* `MultiTaskDataloaders` and `MultiTaskDataloader` fulfill the use cases of `self.dls` and `self.dl` respectively, so `MultiTaskLearner` can reuse the most code from `MyLearner`.

* The `__iter__` of `MultiTaskDataloader`. 
 * The outer `cycle` is for iter through different tasks, and when next `one_batch` we can still use the same iter to iter through different tasks.
 * The inner `cycle` is for getting a batch from that task using `next`. The use of `cycle` make tasks with smaller number of batches will revisit some of batches in an epoch.

* `MultiHeadModel` can be anything that 
 * implement `__getitem__` method, and can return a model for a specific task given that the index of that task.
 * implement `__len__` method, return how many tasks is this multi-task model supports. Just for check.
"""
  
class MultiTaskDataloader():
  def __init__(self, multi_dl, cycle=False): # list of Dataloaders
    self.multi_dl = multi_dl
    self.cycle = cycle

  def __len__(self): return max([len(dl) for dl in self.multi_dl])
  def __iter__(self):
    """
    outer cycle will be controlled by number of tasks
    inner cycle will be controlled by len(MultiTaskDataloader) 
    """
    if self.cycle: return cycle([cycle(dl) for dl in self.multi_dl])
    else: return cycle([not_cycle_infinite(dl) for dl in self.multi_dl])
    
class MultiTaskDataloaders():
  def __init__(self, multi_dls, cycle=False):
    assert all([dls.device==multi_dls[0].device for dls in multi_dls])
    self.multi_dls = multi_dls
    self.train = MultiTaskDataloader([dls.train for dls in self.multi_dls],cycle=cycle)

  def __getitem__(self, i): return self if i is None else self.multi_dls[i]
  def __len__(self): return len(self.multi_dls)
  @property
  def device(self): return self.multi_dls[0].device

class MultiHeadModel(nn.Module):
  def __init__(self, head_less_model, pred_heads):
    super().__init__()
    self.head_less_model = head_less_model
    self.pred_heads = nn.ModuleList(pred_heads)
    self.pred_head_idx = None
  def switch(self, i):
    self.pred_head_idx = i
    return self
  def __len__(self): return len(self.pred_heads)
  def forward(self, inp):
    assert self.pred_head_idx is not None, "`pred_head_idx` is not set."
    return self.pred_heads[self.pred_head_idx](self.head_less_model(inp))

"""
* The core is**`self.current_task_idx`**. Some parts in the whole multi-task training and validating is *task-specific* but some is not. `self.current_task_idx` switch `MultiTaskLeaner` to different status. 
 * `self.current_task_idx` is `None` when in *task-agnostic* part
 * `self.current_task_idx` is i(an int) when in `task-specific` part, furthermore, in i th task.
 * So we in `@property` for `model`, `dls`, `loss_func`, they can detect which task we're dealing with and return thing we need, so we don't need to deal with them when we iterate tasks.

* As to `Callback`s
 * `cbs` is global, will be executed no matter which task or task-agnostic part
 * `multi_cbs` is local, only when task-specific part and is the task *i* will cbs in `multi_cbs[i]` be executed. Note that even callbacks in `multi_cbs` defined for events in task-agnostic part, those won't  be executed.

* Events in *task-specific* part are `begin_batch`, `after_pred` when training, and  all events in validating process (include `begin_batch` and `after_pred` when validating). Other part is *task-agnostic*.
"""

class MultiTaskLearner(MyLearner):
  def __init__(self, multi_dls, multi_model, multi_loss_func=None, task_weights=None, task_names=None, cycle_data=False, cbs=None,  multi_cbs=None, multi_metrics=None, tasks_dont_measure=[],
               opt_func=Adam, lr=defaults.lr, splitter=trainable_params, path=None, model_dir='models', wd=None, wd_bn_bias=False, train_bn=True, moms=(0.95,0.85,0.95)):
    # global task switcher
    self.current_task_idx = None
    self.multi_dls = MultiTaskDataloaders(multi_dls, cycle=cycle_data)
    # store passed arguments
    store_attr(self, "multi_model,task_weights,task_names,multi_cbs,multi_metrics,tasks_dont_measure,opt_func,lr,splitter,model_dir,wd,wd_bn_bias,train_bn,moms")
    # infer
    self.multi_loss_func = self.infer_loss_func(multi_loss_func)
    if task_weights is None: self.task_weights = [1.] * len(self)
    if task_names is None: self.task_names = [f't{i}' for i in range(len(self))]
    if cbs is None: cbs=[]
    if multi_cbs is None: self.multi_cbs = [ [] for _ in range(len(self)) ]
    if multi_metrics is None: self.multi_metrics = [ [] for _ in range(len(self)) ]
    self.path = Path(path) if path is not None else getattr(dls, 'path', Path('.'))
    # check
    assert len(self)==len(self.multi_dls)==len(self.multi_model)==len(self.multi_loss_func)==len(self.task_weights)==len(self.task_names)==len(self.multi_cbs)==len(self.multi_metrics)
    # init
    self.training,self.create_mbar,self.logger,self.opt,self.cbs = False,True,print,None,L()
    self.add_cbs([(cb() if isinstance(cb, type) else cb) for cb in [TrainEvalCallback(),MulitTaskRecorder(),MyMyProgressCallback, *cbs]])
    
  @property
  def device(self):
    assert all([dls.device==self.multi_dls[0].device for dls in self.multi_dls])
    return self.multi_dls[0].device
  @property
  def model(self): return self.multi_model.switch(self.current_task_idx)
  @model.setter
  def model(self, m): self.multi_model = m 
  @property
  def dls(self): return self.multi_dls[self.current_task_idx]
  @property
  def loss_func(self): return self.multi_loss_func[self.current_task_idx]
  
  def infer_loss_func(self, multi_loss_func):
    if multi_loss_func is None:
      multi_loss_func = [ getattr(dls.train_ds, 'loss_func', None) for dls in self.multi_dls]
    else:
      assert len(multi_loss_func) == len(self)
      for i, loss_func in enumerate(multi_loss_func):
        if loss_func is None: # then infer from dls
          multi_loss_func[i] = getattr(self.multi_dls[i].train_ds, 'loss_func', None)
    fails = [ str(i)  for i, loss_func in enumerate(multi_loss_func) if loss_func is None]
    assert not fails, f"Could not infer loss function from the dataloaders for task {', '.join(fails)}, please pass loss functions for them."
    return multi_loss_func
    
  def __len__(self): return len(self.multi_dls)

  def _call_one(self, event_name):
    assert hasattr(event, event_name)
    local_cbs = [] if self.current_task_idx is None else self.multi_cbs[self.current_task_idx]
    with self.added_cbs(local_cbs):
      [cb(event_name) for cb in sort_by_run(self.cbs)]
      
  def one_loss(self):
    if not self.training:
      super().one_loss()
      return 

    losses = []
    for i, dl_iter in zip(range(len(self)), self.dl_iter): # here self.dl_iter is actually dl_iters
      self.current_task_idx = i
      b = next(dl_iter)
      if b is None:
        losses.append(None)
        continue
      self._split(b);                                  self('begin_batch')
      self.pred = self.model(*self.xb);                self('after_pred')
      if len(self.yb) == 0: return
      loss = self.loss_func(self.pred, *self.yb); 
      losses.append(loss)
    self.current_task_idx = None
    self.loss = self.merge_loss(losses);               self('after_loss')
    

  def merge_loss(self, losses):
    # search for first loss that is a tensor to identify dtype and device
    for l in losses:
      if isinstance(l, torch.Tensor): l_dtype,l_device = l.dtype, l.device; break
    # convert all fake loss (0) to tensor
    for i, l in enumerate(losses):
      if l is None: losses[i] = torch.tensor(0,  dtype=l_dtype, device=l_device)
    losses_tensor = torch.stack(losses)
    task_weights_tensor = torch.tensor(self.task_weights, dtype=l_dtype, device=l_device)
    return torch.matmul(losses_tensor,task_weights_tensor)
    
  def _do_epoch_validate(self):
    for self.current_task_idx in range(len(self)):
      if self.current_task_idx in self.tasks_dont_measure: continue
      if self.task_names[self.current_task_idx] in self.tasks_dont_measure: continue
      super()._do_epoch_validate()
    self.current_task_idx = None

"""
#tasks = ['cola', 'sst2', 'mrpc', 'qqp', 'stsb', 'mnli', 'qnli', 'rte', 'wnli']
tasks = ['mrpc','qqp']

dls_s, pred_heads, loss_funcs, metrics_s = [], [], [], []
for task in tasks:
  # multi_dls
  dls_s.append(glue_dls[task])
  # multi_model
  pred_heads.append(SentencePredictHead(electra_config.hidden_size, TARG_VOC_SIZE[task]))    
  # multi_loss_func
  loss_funcs.append(CrossEntropyLossFlat() if task != 'stsb' else MSELossFlat())
  # multi_metrics
  metrics_s.append([eval(f'{metric}()') for metric in METRICS[task]])

multi_head_model = MultiHeadModel(HF_ModelWrapper.from_pretrained(ElectraModel, 'google/electra-small-discriminator', pad_id=hf_tokenizer.pad_token_id, sep_id=hf_tokenizer.sep_token_id), 
                                  pred_heads)

multi_learn = MultiTaskLearner(multi_dls=dls_s,
                               opt_func=partial(Adam, eps=1e-6,),
                               multi_model=multi_head_model,
                               multi_loss_func=loss_funcs,
                               task_weights=[0.5,0.5], # default as [1.] * number of tasks
                               task_names=tasks,
                               multi_metrics=metrics_s,
                               splitter=partial(hf_electra_param_splitter,num_hidden_layers=electra_config.num_hidden_layers,
                                                outlayer_name='pred_heads'),
                               lr=get_layer_lrs(3e-4,0.8,electra_config.num_hidden_layers),
                               )#.to_fp16()

multi_learn.fit(2, 3e-4)
"""