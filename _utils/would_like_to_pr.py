from fastai2.text.all import *

@delegates()
class TextDataloader(TfmdDL):
  def __init__(self, dataset, max_seq_len=float('inf'), sort_by_len=True, agg_mode=None, ignore_gt_maxlen=True, remove_heads=False, remove_tails=False, bos_idx_add=None, eos_idx_add=None, **kwargs):
    super().__init__(dataset, **kwargs)
    assert agg_mode in [None, 'lm', 'lines', 'window']
    assert not (agg_mode and max_seq_len is None) 
    self.sort_by_len = sort_by_len and agg_mode in [None, 'lines'] # sorting makes sense only with these modes
    ignore_gt_maxlen = ignore_gt_maxlen and agg_mode in [None, 'lines'] and max_seq_len is not None
    first_text_tensor = dataset[0][0]
    device, dtype = first_text_tensor.device, first_text_tensor.dtype
    self.bos = torch.tensor([bos_idx_add] if bos_idx_add is not None else [], device=device, dtype=dtype)
    self.eos = torch.tensor([eos_idx_add] if eos_idx_add is not None else [], device=device, dtype=dtype)
    self.add_bos_or_eos = bos_idx_add or eos_idx_add

    store_attr(self,'dataset,max_seq_len,sort_by_len,agg_mode,ignore_gt_maxlen,remove_heads,remove_tails,bos_idx_add,eos_idx_add')
    
    self.samples = L()
    # residual_len will reset to initial_residual_len
    # lm mode: max_seq_len text and 1 right-shift text, so take max_seq_len + 1 window
    self.initial_residual_len = max_seq_len + 1 if agg_mode=='lm' else max_seq_len 
    # keep spaces to add bos to final text 
    if bos_idx_add is not None: self.initial_residual_len -= 1
    if eos_idx_add is not None: self.initial_residual_len -= 1
    self.residual_len, self.new_sample = self.initial_residual_len, []
    # only use [start:end] text to concatenate (if needed)
    self.start = 0 if not remove_heads else 1
    self.end = None if not remove_tails else -1

    for i, sample in enumerate(dataset):
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
    if self.agg_mode in ['lm', 'window']:
      self.samples.shuffle()
    else:
      self.samples.sort(key=lambda s: s[0])
    return idxs

  def cache(self, file_path):
    tmp = self.dataset
    self.dataset = None
    torch.save(self, file_path)
    self.dataset = tmp

  @classmethod
  def from_cache(cls, file_path, dataset):
    dl = torch.load(file_path)
    dl.dataset = dataset
    return dl

  @delegates(TfmdDL.new)
  def new(self, dataset=None, **kwargs):
    return super().new(dataset=dataset,
                       max_seq_len=self.max_seq_len,
                       sort_by_len=self.sort_by_len,
                       agg_mode=self.agg_mode,
                       ignore_gt_maxlen=False, # You can't discard data from valid set, especially test set
                       remove_heads=self.remove_heads,
                       remove_tails=self.remove_tails,
                       bos_idx_add=self.bos_idx_add,
                       eos_idx_add=self.eos_idx_add,
                       **kwargs,)
