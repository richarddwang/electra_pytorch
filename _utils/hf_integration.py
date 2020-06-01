"""
I arranged it from ohmeow/blurr(https://github.com/ohmeow/blurr/blob/67359a8f358b9f044ed561401a720ae5715c63cf/blurr/data.py), because it requires py>=3.7 but Colab has py=3.6 and I simplified and twist it to just for my needs. 

Anyway, checkout ohmeow's fantatistic work !
"""
from functools import partial
from torch import nn
import nlp
from fastai2.text.all import *

class HF_Tokenizer():
    """huggingface friendly tokenization function."""
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
        """Creates a HF_TextBlock via a pandas DataFrame"""

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


"""
Novel huggingface/nlp integration, which mimics `fastai2.data.core.Datasets`
"""
  
class HF_Dataset():

  """ Inheritance by object composition
  I want this class behave like nlp.arrow_dataset.Dataset, and overload some methods. (Inheritance),
  But I don't know how to initialize a nlp.arrow_dataset.Dataset with existing Dataset properly and without additional cost such as a new copy.
  
  So I add every attributes/methods of nlp.arrow_dataset.Dataset, ans pass execution to composed Dataset.
  Notice that __init__, __repr__,__getattribute__,__new__ should'nt be added, when doing this I call it Inheritance by object composition
  ,otherwise it won't work for the reason I don't know.
  """
  for attr_name, attr in nlp.arrow_dataset.Dataset.__dict__.items():
    if attr_name not in ['__init__', '__repr__','__getattribute__','__new__'] + ['__getitem__','__iter__',]:
      if callable(attr): exec(f'def {attr_name}(self,*args,**kwargs): return self.dataset.{attr_name}(*args,**kwargs)')
      else: exec(f'@property\ndef {attr_name}(self): return self.dataset.{attr_name}')

  def __init__(self, dataset, cols, encode_types, decode_funcs, decode_types):
    store_attr(self, 'dataset,cols,encode_types,decode_funcs,decode_types')

  def __getitem__(self, i):
    sample = self.dataset[i]
    return tuple( enc_type(sample[col]) for col, enc_type in zip(self.cols, self.encode_types))

  def __iter__(self):
    """
    default __iter__ will iter until get IndexError, 
    but ArrowDataset gives you ValueError when out of index.
    So we have to explicitly define __iter__ method
    """
    for i in range(len(self)): yield self[i] 

  def decode(self, o, full=True): return tuple( de_type(de_fc(o_)) for o_,de_fc,de_type in zip(o,self.decode_funcs,self.decode_types))
  #def __len__(self): return len(self.dataset)

class HF_Datasets(FilteredBase):
  def __init__(self, datasets, cols, encode_types, decode_funcs, decode_types):
    assert len(cols) == len(decode_funcs) == len(encode_types) == len(decode_types) == len(decode_funcs)
    for ds in datasets: ds.set_format(type='torch', columns=cols)
    self.datasets = L(HF_Dataset(ds, cols, encode_types, decode_funcs, decode_types) for ds in datasets)
  def subset(self, i): return self.datasets[i]
  def __getitem__(self, i): return self.datasets[i]
  @property
  def n_subsets(self): return len(self.datasets)
