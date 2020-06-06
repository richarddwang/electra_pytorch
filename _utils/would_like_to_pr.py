from tqdm import tqdm
import sklearn.metrics as skm
import scipy.stats as scs
from fastai2.text.all import *

@delegates()
class TextDataloader(TfmdDL):
  def __init__(self, dataset, max_seq_len=float('inf'), sort_by_len='desc', agg_mode=None, ignore_gt_maxlen=False, remove_heads=False, remove_tails=False, bos_idx_add=None, eos_idx_add=None, samples=None, **kwargs):
    super().__init__(dataset, **kwargs)
    assert agg_mode in [None, 'lm', 'lines', 'window']
    assert not (agg_mode and max_seq_len is None)
    assert sort_by_len in [False, 'desc', 'asc']
    if agg_mode in ['window','lm']: sort_by_len=False # sorting makes no sense with these modes
    ignore_gt_maxlen = ignore_gt_maxlen and agg_mode in [None, 'lines'] and max_seq_len is not None
    first_text_tensor = dataset[0][0]
    device, dtype = first_text_tensor.device, first_text_tensor.dtype
    self.bos = torch.tensor([bos_idx_add] if bos_idx_add is not None else [], device=device, dtype=dtype)
    self.eos = torch.tensor([eos_idx_add] if eos_idx_add is not None else [], device=device, dtype=dtype)
    self.add_bos_or_eos = bos_idx_add or eos_idx_add
    # only use [start:end] text to concatenate (if needed)
    self.start = 0 if not remove_heads else 1
    self.end = None if not remove_tails else -1

    store_attr(self,'dataset,max_seq_len,sort_by_len,agg_mode,ignore_gt_maxlen,remove_heads,remove_tails,bos_idx_add,eos_idx_add')
    
    if samples is not None: # Load from cache
      if sort_by_len: self.samples = sorted(samples, key=lambda s: s[0], reverse=True if sort_by_len=='desc' else False)
      else: self.samples = samples
      self.n = len(samples)
      return

    self.samples = L()
    # residual_len will reset to initial_residual_len
    # lm mode: max_seq_len text and 1 right-shift text, so take max_seq_len + 1 window
    self.initial_residual_len = max_seq_len + 1 if agg_mode=='lm' else max_seq_len 
    # keep spaces to add bos to final text 
    if bos_idx_add is not None: self.initial_residual_len -= 1
    if eos_idx_add is not None: self.initial_residual_len -= 1
    self.residual_len, self.new_sample = self.initial_residual_len, []

    for i, sample in tqdm(enumerate(dataset), desc='TextDataloader init:', total=len(dataset), leave=False):
      line_len = len(sample[0])
      if remove_heads: line_len -= 1
      if remove_tails: line_len -= 1
      
      if max_seq_len is not None and line_len > self.initial_residual_len and agg_mode in [None, 'lines']:
        if ignore_gt_maxlen: continue
        else: raise ValueError(f'The {i} th text line in dataset has length {line_len}(without removing head or tail, {len(sample[0])}), and is longer than max length {self.initial_residual_len}(without add bos or eos, {max_seq_len})')
        
      if agg_mode is None: self.samples.append( (line_len, i) )
      elif agg_mode == 'lines': self._accumulate_lines(i, line_len)
      else: self._accumulate_window(i, line_len)
    
    if agg_mode is not None and self.new_sample:
      if agg_mode == 'lines': self.samples.append((self.max_seq_len-self.residual_len, self.new_sample))
      else: self.samples.append(self.new_sample)

    # sort if needed
    if sort_by_len:
      self.samples.sort(key=lambda s: s[0], reverse=True if sort_by_len=='desc' else False)
    # specify total number of samples
    self.n = len(self.samples)
      
  def _accumulate_lines(self, i, line_len):
    if line_len <= self.residual_len:
      self.new_sample.append(i)
      self.residual_len -= line_len
    else:
      self.samples.append((self.max_seq_len-self.residual_len, self.new_sample))
      self.new_sample = [i]
      self.residual_len = self.initial_residual_len - line_len

  def _accumulate_window(self, i, line_len):
    usable_len = line_len
    cursor = self.start
    while usable_len != 0:
      use_len = min(usable_len, self.residual_len)
      self.new_sample.append((i, cursor, cursor+use_len))
      self.residual_len -= use_len
      usable_len -= use_len
      cursor += use_len
      if self.residual_len == 0:
        self.samples.append(self.new_sample)
        self.new_sample = []
        self.residual_len = self.initial_residual_len

  def create_item(self, s):
    if self.agg_mode is None:
      "samples = [ (length, idx), ... ]"
      idx = self.samples[s][1]
      sample = self.dataset[idx]
      line = sample[0][self.start:self.end]
      text = torch.cat([self.bos, line, self.eos]) if self.add_bos_or_eos else line
      return ( TensorText(text), *sample[1:] )
    elif self.agg_mode == 'lines':
      "samples = [ (length, [idx, idx, ...]) , ... ]"
      agg = [ self.dataset[idx][0][self.start:self.end] for idx in self.samples[s][1] ]
      agg_text = concat(self.bos, *agg, self.eos) if self.add_bos_or_eos else concat(*agg)
      return (TensorText(agg_text), )
    else: # window or lm
      "samples = [ (idx,start,end) ]"
      agg = [ self.dataset[idx][0][start:end] for idx,start,end in self.samples[s] ]
      agg_text = concat(self.bos, *agg, self.eos) if self.add_bos_or_eos else concat(*agg)
      if self.agg_mode == 'window':
        return (TensorText(agg_text), )
      else: # 'lm'
        return (LMTensorText(agg_text[:-1]), TensorText(agg_text[1:]))

  def shuffle_fn(self, idxs):
    if not self.sort_by_len: # notice sort_by_len in lm and winodw mode will be False
      self.samples.shuffle()
    return idxs

  def desc_sort(self):
    assert self.agg_mode not in ['window','lm'], f"Sorting by length makes no sense on aggregation mode {self.agg_mode}"
    self.samples.sort(key=lambda s: s[0], reverse=True)
    self.sort_by_len = 'desc'

  def asc_sort(self):
    assert self.agg_mode not in ['window','lm'], f"Sorting by length makes no sense on aggregation mode {self.agg_mode}"
    self.samples.sort(key=lambda s: s[0], reverse=False)
    self.sort_by_len = 'asc'

  def cache(self, file_path):
    torch.save(self, file_path)

  def __getstate__(self):
    "specify something you don't want pickle here, remember to use copy to not modfiy orginal instance"
    state = self.__dict__.copy()
    state['dataset'] = None
    return state

  #@delegates(TextDataloader.__init__) but we haven't evaluated TextDataloader
  @delegates(TfmdDL.new)
  @classmethod
  def from_cache(cls, file_path, dataset, **kwargs):
    dl = torch.load(file_path)
    dl.dataset = dataset

    # Reject change that cause arguments be inconsistent with loaded `self.samples` record 
    for arg in ['max_seq_len','agg_mode','ignore_gt_maxlen','remove_heads','remove_tails']:
      assert arg not in kwargs, f"Specifying {arg} will make it inconsistent with cached internal record."
    if 'sort_by_len' in kwargs:
      assert not (dl.sort_by_len and not kwargs['sort_by_len']), f"Cached textdl is internal sorted, it can't restore orignal order."
    for arg in ['bos_idx_add','eos_idx_add']:
      if arg in kwargs: assert (kwargs[arg] is None) == (getattr(dl, arg) is None), f"You can't change whether to add head/eos from cached setting."
    # TextDataloader.new guess creating validation dataloader if don't drop_last, but it might not be the case
    kwargs['ignore_gt_maxlen'] = dl.ignore_gt_maxlen
    # Even if spefify no kwargs and just load original dataloader, using new method can update device and dtype of bos and eos for this dataset
    dl = dl.new(dataset, samples=dl.samples, **kwargs)
    # Consider whether setting up newly pased batch tfms, cuz new method just make do_setup=false
    # Actually I don't know if it is good, but at leaat it works for pad_input_chunk as before_batch
    if kwargs.pop('do_setup', True):
      for nm in ['after_item','before_batch','after_batch']:
        if nm in kwargs:
          kwargs[nm] = Pipeline(kwargs.get(nm,None)) # don't know why Pipeline creating in TfmdDL won't be done in this case, but we can do it here and even it has done, it is ok we just Pipeline it again. 
          pv(f"Setting up {nm}: {kwargs[nm]}", kwargs.pop('verbose', False))
          kwargs[nm].setup(dl)
    return dl

  @delegates(TfmdDL.new)
  def new(self, dataset=None, **kwargs):
    cur_args = dict(max_seq_len=self.max_seq_len, sort_by_len=self.sort_by_len,agg_mode=self.agg_mode,ignore_gt_maxlen=self.ignore_gt_maxlen,remove_heads=self.remove_heads, remove_tails=self.remove_tails, bos_idx_add=self.bos_idx_add, eos_idx_add=self.eos_idx_add)
    
    # we assume if you don't drop_last, you are going to create validation dl, specify ignore_gt_maxlen in kwargs to overwrite it if this is not in the case  
    if not getattr(kwargs, 'drop_last', self.drop_last): 
      cur_args['ignore_gt_maxlen'] = False # You can't discard data from dataset for validation, especially test set
    
    return super().new(dataset=dataset,
                       **merge(cur_args, kwargs)) # kwargs overwrite cur_args

"""
There is bug in scikit_learn (https://github.com/scikit-learn/scikit-learn/issues/16924)
so you always get `RuntimeWarning: invalid value encountered in double_scalar` and get a value 0.0
"""
def MatthewsCorrCoef(sample_weight=None, **kwargs):
    return skm_to_fastai(skm.matthews_corrcoef, sample_weight=sample_weight, **kwargs)

"""
If you see PearsonRConstantInputWarning, that may mean the model always outputs a specific label,
which is probable when you test something with very small dataset size or very short training
"""
@delegates(AccumMetric.__init__)
def scs_to_fastai(func, dim_argmax=-1, **kwargs):
  return AccumMetric(func, dim_argmax=dim_argmax, **kwargs)

@delegates(scs_to_fastai)
def PearsonCorrCoef(**kwargs):
    "Pearson correlation coefficient for regression problem"
    def pearsonr(x,y): return scs.pearsonr(x,y)[0]
    return scs_to_fastai(pearsonr, invert_arg=False, dim_argmax=None, **kwargs)
"""
For the same reason as Pearson correlation, you may see nan for value of Spearman correlation, and
RuntimeWarning: invalid value encountered in true_divide, because there is no mutual change. 
(see https://stackoverflow.com/questions/45897003/python-numpy-corrcoef-runtimewarning-invalid-value-encountered-in-true-divide)
"""
# metric for STS task
@delegates(scs_to_fastai)
def SpearmanCorrCoef(axis=0, nan_policy='propagate', **kwargs):
    "Spearman correlation coefficient for regression problem"
    def spearmanr(a,b=None,**kwargs): return scs.spearmanr(a,b,**kwargs)[0]
    return scs_to_fastai(spearmanr, invert_arg=False, dim_argmax=None, axis=axis, nan_policy=nan_policy, **kwargs)

"""
I would like more uniform way to pass the metrics, no matter loss_func or metric,
instantiate it and then pass.
This uniform way also make it possible such as `metrics=[m() for m inTASK_METRICS[task]]`
"""
def Accuracy(axis=-1):
  return AvgMetric(partial(accuracy, axis=axis))


@log_args
@delegates(keep=True)
class LabelSmoothingCrossEntropyFlat(BaseLoss):
    "Same as `nn.CrossEntropyLoss`, but flattens input and target."
    y_int = True
    def __init__(self, *args, axis=-1, **kwargs): super().__init__(LabelSmoothingCrossEntropy, *args, axis=axis, **kwargs)
    def activation(self, out): return F.softmax(out, dim=-1)
    def decodes(self, out):    return out.argmax(dim=-1)
