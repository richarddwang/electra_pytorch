from functools import partial
from tqdm import tqdm
import pyarrow as pa
from torch import nn
import nlp
from fastai2.text.all import *

class HF_TokenizeTfm():
  """
  Args:
    `hf_dset` (`nlp.Dataset`)
    `cols`: with one of the following signature:
      - `cols`(`List[str]`): tokenize the col into col
      - `cols`(`Dict[str]`): tokenize the col(key) and into col(value)
    `hf_tokenizer`: tokenizer of HuggingFace/Transformers.
    `remove_original`: after tokenization, remove all original columns to save cache size.
  """
  def __init__(self, hf_dset, cols, hf_tokenizer, remove_original=False):
    if isinstance(cols, list): cols = {c:c for c in cols}
    assert isinstance(cols, dict)
    self.hf_dset, self.cols, self.tokenizer = hf_dset, cols, hf_tokenizer
    self.remove_original = remove_original
    if remove_original:
      for in_col,out_col in cols.items(): assert in_col !=  out_col
    """
    If don't specify cache file name, it will be hashed binary of pickled function that
    passed to `map`, so if you pass the same function, it knows to use cache.
    But tokenizer can't be pickled, so use tokenizer config to make tfms use different 
    tokenizers unique.  
    """
    self.tokenizer_config = hf_tokenizer.pretrained_init_configuration
  
  def __call__(self, example):
    for in_col, out_col in self.cols.items():
      example[out_col] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(example[in_col]))
    return example

  def __getstate__(self):
    "specify something you don't want pickle here, remember to use copy to not modfiy orginal instance"
    state = self.__dict__.copy() 
    state['tokenizer'] = None 
    return state

  def map(self, **kwargs):
    if self.remove_original:
      assert 'remove_columns' not in kwargs, "You have specified to remove all original columns."
      return self.hf_dset.map(self, remove_columns=self.hf_dset.column_names, **kwargs)
    else:
      return self.hf_dset.map(self, **kwargs)

@delegates()
class HF_Dataloader(TfmdDL):
  """
  Args:
    `dataset`: any class that output a tuple, which has token ids in its first element, from `__getitem__`.
    `pad_idx` (`int`): If sepcified, pad texts to the longest text in the batch.
    `sort` (`Optional[bool]`, default: `True`): Sort the samples with their length, thus samples of similar legnth collated into a batch and we can pad less. Notice if it is True, the shuffle will be overrided and not shuffle.
    `filterout` (`Optional[callable(*args) -> bool]`, , default: `None`): if not `None`, judege whether exclude this sample with this sample(`tuple`) as args
    `cache_file` (`Optional[str]`, default: `None`): A name of json file to store the computed record of results of sort or filterout.   
  """
  def __init__(self, dataset, pad_idx, sort=True, filterout=None, cache_file=None, **kwargs):
    if pad_idx is not None:
      kwargs['before_batch'] = partial(pad_input_chunk, pad_idx=pad_idx, pad_first=False)

    cache_file = Path(cache_file) if cache_file else None
    if cache_file and cache_file.exists():
      with cache_file.open(mode='r') as f: self.samples = json.load(f)
    elif sort or filterout:
      if cache_file: cache_file.touch()
      try:
        if filterout is None: filterout = lambda *args: False
        self.samples = [ (i,len(sample[0])) for i, sample in tqdm(enumerate(dataset), leave=False) if not filterout(*sample) ]
        if sort: self.samples.sort(key=lambda t:t[1], reverse=True)
      except Exception as e:
        os.remove(cache_file)
        raise e
      if cache_file:
        with cache_file.open(mode='w') as f: json.dump(self.samples, f)
    else:
      self.samples = [ (i,None) for i in range(len(dataset))]

    store_attr(self, 'pad_idx,sort,filterout,cache_file')
    super().__init__(dataset, **kwargs)
    if sort: self.shuffle=False
    self.n = len(self.samples)
  
  def create_item(self, i): return self.dataset[self.samples[i][0]]

  def new(self, dataset, **kwargs):
    return super().new(dataset=dataset, pad_idx=self.pad_idx, sort=self.sort, filterout=self.filterout, **kwargs)

class HF_Dataset(FilteredBase):
  """
  Args:
    `hf_dset` (`nlp.Dataset`)
    `cols`: with one of the following signature:
    - `cols`(`List[str]`): 
      - if of length 1, regard the 1st element as text
      - if of length 2, regrad the 1st element as text, 2nd as category
    - `cols`(`Dict[Fastai2 Semantic Tesor]`): {`inp_col`:tensor type}: output sample as tuple of values of `inp_col` in order, and encode/decode with the tensor type,
    `hf_tokenizer`: tokenizer of HuggingFace/Transformers
    `pretty_show` (`Optional[bool]`, default:`False`): Show the original sentence instead of tokens.
  """
  def __init__(self, hf_dset, cols, hf_tokenizer=None, pretty_show=False, n_inp=1):
    
    # some default setting for tensor type used in decoding
    if isinstance(cols, list): 
      if n_inp==1: 
        if len(cols)==1: cols = {cols[0]: TensorText}
        elif len(cols)==2: cols = {cols[0]: TensorText, cols[1]: TensorCategory}
      else: cols = { c: noop for c in cols }
    assert isinstance(cols, dict)
    
    # make dataset output pytorch tensor
    if hf_dset.format['type'] != 'torch': 
      hf_dset.set_format( type='torch', columns=list(cols.keys()) )

    # store attributes
    store_attr(self, "hf_dset,cols,n_inp,hf_tokenizer,pretty_show")

  def __getitem__(self, idx):
    sample = self.hf_dset[idx]
    return tuple( tensor_cls(sample[col]) for col, tensor_cls in self.cols.items() )

  def __len__(self): return len(self.hf_dset)

  @property
  def column_names(self): return list(self.cols.keys())

  def decode(self, o, full=True): 
    return tuple( self._decode(o_) for o_ in o )

  @typedispatch
  def _decode(self, t:torch.Tensor): assert False, "You didn't specify a tensor type, thus not be able to decode and show."

  @typedispatch
  def _decode(self, t:TensorText): 
    assert self.hf_tokenizer, "You should give huggingface tokenizer if you want to show batch."
    if self.pretty_show: text = self.hf_tokenizer.decode([idx for idx in t if idx != self.hf_tokenizer.pad_token_id])
    else: text = ' '.join(self.hf_tokenizer.convert_ids_to_tokens(t))
    return TitledStr(text)

  @typedispatch
  def _decode(self, t:LMTensorText): return self._decode[TensorText](self, t)

  @typedispatch
  def _decode(self, t:TensorCategory): return Category(t.item())
  
class HF_Datasets(FilteredBase):
  _dl_type,_dbunch_type = HF_Dataloader,DataLoaders
  def __init__(self, hs_dsets: dict, *args, **kwargs):
    """
    Args:
      `hs_dsets` (`Dict[nlp.Dataset]`): the order of dict items will be the order of `HF_Dataloader`s  
    """
    self.hs_dsets = { split: HF_Dataset(dset, *args, **kwargs) for split, dset in hs_dsets.items()}
  def subset(self, i): return list(self.hs_dsets.values())[i]
  def __getitem__(self, split): return self.hs_dsets[split]
  @property
  def n_subsets(self): return len(self.hs_dsets)
  def dataloaders(self, *args, cache_files=None, device='cpu', **kwargs):
    """
    Args:
      `*args, **kwargs`: `for FilteredBase.dataloaders`
      `cache_files` (`Optional[str]`, default:`None`): cache file names for `HF_Dataloader`s
      `device` (`Optional[str]`, default:`'cpu'`): cuz will read a batch for test when creating `Dataloader`, so I set the default device to cpu to less the memory burden of cuda:0 
    """
    dl_kwargs = kwargs.pop('dl_kwargs', [{} for _ in range(len(self.hs_dsets))])
    if cache_files:
      assert len(cache_files) == len(self.hs_dsets)
      for i, dl_kwarg in enumerate(dl_kwargs): dl_kwarg['cache_file'] = cache_files[i]
    return super().dataloaders(*args, dl_kwargs=dl_kwargs, device=device, **kwargs)

class AggregateTransform():
  """
  Inherit this class and implement `accumulate` and `create_example`
  """
  def __init__(self, hf_dset, inp_cols, out_cols, init_attrs, drop_last=False):
    """
    Args:
      `hf_dset` (`nlp.Dataset`)
      `inp_cols` (`List[str]`)
      `out_cols` (`List[str]`)
      `init_attrs` (`List[str]`): name of attributes of children class that need to be their initial status when starts to aggregate dataset. i.e. Those defined in `__init__` and the value will changed during `accumulate`
      `drop_last` (`Optional[bool]`, default: `False`): whether to drop the last accumulated sample.
    """
    self.hf_dset = hf_dset
    self.inp_cols, self.out_cols =  inp_cols, out_cols
    # batched map need dataset be in python format
    hf_dset.set_format(type=None, columns=inp_cols)
    # dealing with last sample
    self.last_idx = len(hf_dset) - 1
    self.drop_last = drop_last
    # for reset
    self.init_attrs = init_attrs
    self.original_vals = [deepcopy(getattr(self, attr)) for attr in init_attrs]  

  def __call__(self, b, indices):
    # `nlp.Dataset.map` first test with several samples which affects our attrs, so we need to reinitialize.
    if 0 in indices: # reset
      for attr,val in zip(self.init_attrs, self.original_vals): setattr(self, attr, deepcopy(val))

    self.new_b = { c:[] for c in self.out_cols }
    for z in zip(*b.values()):
      self.accumulate(*z)
    
    # whehther put last example when it is last batch of `map`
    if not self.drop_last and self.last_idx in indices: 
      self.commit_example(self.create_example())

    return self.new_b

  def commit_example(self, example):
    if example is None: return
    for col,val in example.items():
      self.new_b[col].append(val) 

  def accumulate(self, *args):
    """
    Given a example, do `self.commit_example(self.create_example()) when a new aggregated sample is ready.`
    Args:
      `args`: nlp.Dataset[i][inp_col] for inp_col in self.inp_cols
    """ 
    raise NotImplementedError
  
  def create_example(self): 
    """
    When it is ready, create a sample (Dict[Any])
    """
    raise NotImplementedError

  def map(self, batch_size=1000, test_batch_size=20, **kwargs):
    """
    `batch_size`: see `nlp.Dataset.map`
    `test_batch_size` (`int`, default=`20`): we infer the new schema of the aggregated dataset by the outputs of testing that passed first `test_batch_size` samples to aggregate. Depending how many sample aggreagted can you have a sample, this number might need to be higher.
    """
    test_inputs, test_indices = self.hf_dset[:test_batch_size], list(range(test_batch_size))
    test_output = self(test_inputs,test_indices)
    for col,val in test_output.items(): assert val, f"Didn't get any example in test, you might want to try larger `test_batch_size` than {test_batch_size}"
    assert sorted(self.out_cols) == sorted(test_output.keys()), f"Output columns are {self.out_cols}, but get example with {list(test_output.keys())}"
    arrow_schema = pa.Table.from_pydict(test_output).schema
    return self.hf_dset.map(function=self, batched=True, batch_size=batch_size, with_indices=True,
                            arrow_schema=arrow_schema, **kwargs)

class LMTransform(AggregateTransform):
  def __init__(self, hf_dset, max_len, text_col, x_text_col='x_text', y_text_col='y_text', **kwargs):
    self.text_col, self.x_text_col, self.y_text_col = text_col, x_text_col, y_text_col
    self._max_len = max_len + 1
    self.residual_len, self.new_text = self._max_len, []
    super().__init__(hf_dset, inp_cols=[text_col], out_cols=[x_text_col, y_text_col], init_attrs=['residual_len', 'new_text'], **kwargs)
    

  def accumulate(self, text): # *inp_cols
    "text: a list of indices"
    usable_len = len(text)
    cursor = 0
    while usable_len != 0:
      use_len = min(usable_len, self.residual_len)
      self.new_text += text[cursor:cursor+use_len]
      self.residual_len -= use_len
      usable_len -= use_len
      cursor += use_len
      if self.residual_len == 0:
        self.commit_example(self.create_example())   

  def create_example(self):
    # when read all data, the accumulated new_text might be less than two characters.
    if len(self.new_text) >= 2: 
      example = {self.x_text_col:self.new_text[:-1], self.y_text_col:self.new_text[1:]}
    else:
      example = None # mark "don't commit this"
    # reset accumulators
    self.new_text = []
    self.residual_len = self._max_len

    return example

# To just take hidden features output
class HF_ModelWrapper(nn.Module):
  
  @classmethod
  def from_pretrained(cls, hf_cls, cp_name, pad_id, sep_id=None):
    return cls(model=hf_cls.from_pretrained(cp_name), pad_id=pad_id, sep_id=sep_id)
  
  def __init__(self, model, pad_id, sep_id=None):
    "pass sep token id if sentence A sentence B setting. (default is sentence A setting)"
    super().__init__()
    self.model = model
    self.pad_id, self.sep_id = pad_id, sep_id
    
  def forward(self, x):
    attn_mask = x!= self.pad_id
    if self.sep_id is None:
      return self.model(x, attn_mask)[0]
    else:
      return self.model(x, attn_mask, token_type_ids=self._token_type_ids_for(x))[0]

  def _token_type_ids_for(self, x):
    "x: (batch size, sequence length)"
    num_sep = (x==self.sep_id).sum().item()
    if num_sep == x.shape[0]: return None
    assert num_sep == 2*x.shape[0], "Samples should all contains only one or all contains only two [SEP] in each of their texts"
    tok_type_ids = torch.zeros(x.shape, dtype=torch.long, device=x.device)
    second_sent_head_pos = [ s.tolist().index(self.sep_id)+1 for s in x]
    # [[CLS, I, am, hero, SEP, Yes, I, am, SEP],...] -> [5,..]
    tok_type_ids[[torch.arange(x.shape[0]),second_sent_head_pos]] = 1
    # tok_type_ids == [[0,0,..,0,1,0,0,..,0], ...]
    return tok_type_ids.cumsum(dim=1)
    # tok_type_ids.cumsum(dim=1) == [[0,0,..,0,1,1,..,1], ...]

"""
Below is arranged from ohmeow/blurr(https://github.com/ohmeow/blurr/blob/67359a8f358b9f044ed561401a720ae5715c63cf/blurr/data.py), because it requires py>=3.7 but Colab has py=3.6 and I simplified and twist it to just for my needs. 
Anyway, checkout ohmeow's fantatistic work !
"""
"""
class HF_Tokenizer():
    "huggingface friendly tokenization function."
    def __init__(self, hf_tokenizer, **kwargs):
        self.hf_tokenizer = hf_tokenizer
    def __call__(self, items):
        for txt in items: yield self._tokenize(txt)
    def _tokenize(self, txt):
        return [self.hf_tokenizer.cls_token,*self.hf_tokenizer.tokenize(txt), self.hf_tokenizer.sep_token]

class HF_TextBlock(TransformBlock):
    @delegates(Numericalize.__init__)
    def __init__(self, tok_tfm, hf_tokenizer, task=None,
                 hf_batch_tfm=None, vocab=None, max_seq_len=512, **kwargs):
      return super().__init__(type_tfms=[tok_tfm, Numericalize(vocab, **kwargs)],
                              dl_type=SortedDL,
                              dls_kwargs={ 'before_batch': partial(pad_input_chunk, 
                                                                   pad_idx=hf_tokenizer.pad_token_id,
                                                                   pad_first=False) })
    @classmethod
    @delegates(Tokenizer.from_df, keep=True)
    def from_df(cls, text_cols, hf_tokenizer, task=None,
                res_col_name='text', vocab=None, hf_batch_tfm=None, max_seq_len=512, **kwargs):
        "Creates a HF_TextBlock via a pandas DataFrame"

        # grab hf tokenizer class to do the actual tokenization (via tok_func) and its vocab
        tokenizer_cls = partial(HF_Tokenizer, hf_tokenizer=hf_tokenizer)
        if (vocab is None): vocab = list(hf_tokenizer.get_vocab())

        rules = kwargs.pop('rules', [] )
        tok_tfm = Tokenizer.from_df(text_cols,
                                    res_col_name=res_col_name,
                                    tok_func=tokenizer_cls,
                                    rules=rules, **kwargs)

        return cls(tok_tfm, hf_tokenizer=hf_tokenizer, task=task,
                   hf_batch_tfm=hf_batch_tfm, vocab=vocab, max_seq_len=max_seq_len)
"""



class HF_MergedDataset():
  def __init__(self, *datasets):
    self.dsets = datasets
  def __len__(self):
    return reduce(lambda a,d: a+len(d), self.dsets, 0)
  def __getitem__(self, i):
    for dset in self.dsets:
      if i < len(dset): return dset[i]
      else: i -= len(dset)
  def set_format(self, type, columns):
    for dset in self.dsets: dset.set_format(type, columns)
  @property
  def cache_files(self):
    return concat(*[ds.cache_files for ds in self.dsets])