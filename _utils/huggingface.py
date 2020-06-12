from functools import partial
from tqdm import tqdm
import pyarrow as pa
from torch import nn
import nlp
from fastai2.text.all import *

class HF_TokenizeTfm():
  
  def __init__(self, hf_dset, cols, hf_tokenizer, remove_original=False):
    if isinstance(cols, list): cols = {c:c for c in cols}
    assert isinstance(cols, dict)
    self.hf_dset, self.cols, self.tokenizer = hf_dset, cols, hf_tokenizer
    self.remove_original = remove_original
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
  
  def __init__(self, dataset, pad_idx, sort=True, **kwargs):
    if pad_idx is not None:
      kwargs['before_batch'] = partial(pad_input_chunk, pad_idx=pad_idx, pad_first=False)
    if sort:
      self.lens = [ len(sample[0]) for sample in dataset ]
    store_attr(self, 'pad_idx,sort')
    super().__init__(dataset, **kwargs)
  
  def get_idxs(self):
    idxs = super().get_idxs()
    if not self.sort : return idxs
    return sorted(idxs, key=lambda i: self.lens[i], reverse=True)

  def new(self, dataset, **kwargs):
    return super().new(dataset=self.dataset, pad_idx=self.pad_idx, sort=self.sort, **kwargs)

class HF_Dataset(FilteredBase):
  
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
    self.hs_dsets = { split: HF_Dataset(dset, *args, **kwargs) for split, dset in hs_dsets.items()}
  def subset(self, i): return list(self.hs_dsets.values())[i]
  def __getitem__(self, split): return self.hs_dsets[split]
  @property
  def n_subsets(self): return len(self.hs_dsets)

class AggregateTransform():
  def __init__(self, hf_dset, inp_cols, out_cols, init_attrs, drop_last=False):
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
    for z in tqdm(list(zip(*b.values())), leave=False):
      self.accumulate(*z)
    
    # whehther put last example when it is last batch of `map`
    if not self.drop_last and self.last_idx in indices: 
      self.commit_example(self.create_example())

    return self.new_b

  def commit_example(self, example):
    if example is None: return
    for col,val in example.items():
      self.new_b[col].append(val) 

  def accumulate(self, *args): raise NotImplementedError
  def create_example(self): raise NotImplementedError

  def map(self, batch_size=1000, test_batch_size=20, **kwargs):
    test_inputs, test_indices = self.hf_dset[:test_batch_size], list(range(test_batch_size))
    test_output = self(test_inputs,test_indices)
    for col,val in test_output.items(): assert val, f"Didn't get any example in test, you might want to try larger `test_batch_size` than {test_batch_size}"
    assert sorted(self.out_cols) == sorted(test_output.keys()), f"Output columns are {self.out_cols}, but get example with {list(test_output.keys())}"
    arrow_schema = pa.Table.from_pydict(test_output).schema
    return self.hf_dset.map(function=self, batched=True, batch_size=batch_size, with_indices=True,
                            arrow_schema=arrow_schema, **kwargs)

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