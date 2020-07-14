from functools import partial
from pathlib import Path
import json
from tqdm import tqdm
import pyarrow as pa
from torch import nn
import nlp
from fastai2.text.all import *

@delegates()
class MySortedDL(TfmdDL):
    "A `DataLoader` that goes throught the item in the order given by `sort_key`"

    def __init__(self, dataset, pad_idx, srtkey_fc=None, filter_fc=False, cache_file=None, **kwargs):
        """
        Args:
            `dataset`:
            `srtkey_fc` (Callable[*args]->int): Get elements of a sample and return a sorting key. `None` for getting the length of first element, `False` to not sort 
            `filter_fc` (Callable[*args]->bool): Get elements of a sample and return `False` to filter out the sample
            `cache_file` (`Optional[str]`, default: `None`): Path of a json file to store the computed record of results of sort or filterout.
        """
        # Defaults
        if pad_idx is not None: kwargs['before_batch'] = partial(pad_input_chunk, pad_idx=pad_idx, pad_first=False)
        if srtkey_fc is not False: srtkey_fc = lambda *x: len(x[0])
        cache_file = Path(cache_file) if cache_file else None
        idmap = list(range(len(dataset)))
        
        # Save attributes
        super().__init__(dataset, **kwargs)
        store_attr(self, 'pad_idx,srtkey_fc,filter_fc,cache_file,idmap')

        # Prepare records for sorting / filtered samples
        if srtkey_fc or filter_fc:
          if cache_file and cache_file.exists():
            # load cache and check
            with cache_file.open(mode='r') as f: cache = json.load(f)
            idmap, srtkeys = cache['idmap'], cache['srtkeys']
            if srtkey_fc: 
              assert srtkeys, "srtkey_fc is passed, but it seems you didn't sort samples when creating cache."
              self.srtkeys = srtkeys
            if filter_fc:
              assert idmap, "filter_fc is passed, but it seems you didn't filter samples when creating cache."
              self.idmap = idmap
          else:
            # overwrite idmap if filter, get sorting keys if sort
            idmap = []; srtkeys = []
            for i in tqdm(range_of(dataset), leave=False):
                sample = self.do_item(i)
                if filter_fc and not filter_fc(*sample): continue
                if filter_fc: idmap.append(i)
                if srtkey_fc: srtkeys.append(srtkey_fc(*sample))
            if filter_fc: self.idmap = idmap
            if srtkey_fc: self.srtkeys = srtkeys
            # save to cache
            if cache_file:
              try: 
                with cache_file.open(mode='w+') as f: json.dump({'idmap':idmap,'srtkeys':srtkeys}, f)
              except: os.remove(str(cache_file))
          # an info for sorting
          if srtkey_fc: self.idx_max = np.argmax(self.srtkeys)
          # update number of samples
          if filter_fc: self.n = self.n = len(self.idmap)

    def create_item(self, i): return self.dataset[self.idmap[i]]

    def get_idxs(self):
        idxs = super().get_idxs()
        if self.shuffle: return idxs
        if self.srtkey_fc: return sorted(idxs, key=lambda i: self.srtkeys[i], reverse=True)
        return idxs

    def shuffle_fn(self,idxs):
        if not self.srtkey_fc: return super().shuffle_fn(idxs)
        idxs = np.random.permutation(self.n)
        idx_max = np.where(idxs==self.idx_max)[0][0]
        idxs[0],idxs[idx_max] = idxs[idx_max],idxs[0]
        sz = self.bs*50
        chunks = [idxs[i:i+sz] for i in range(0, len(idxs), sz)]
        chunks = [sorted(s, key=lambda i: self.srtkeys[i], reverse=True) for s in chunks]
        sort_idx = np.concatenate(chunks)

        sz = self.bs
        batches = [sort_idx[i:i+sz] for i in range(0, len(sort_idx), sz)]
        sort_idx = np.concatenate(np.random.permutation(batches[1:-1])) if len(batches) > 2 else np.array([],dtype=np.int)
        sort_idx = np.concatenate((batches[0], sort_idx) if len(batches)==1 else (batches[0], sort_idx, batches[-1]))
        return iter(sort_idx)

    @delegates(TfmdDL.new)
    def new(self, dataset=None, **kwargs):
        # We don't use filter_fc here cuz we can't don't validate certaion samples in dev/test set. 
        return super().new(dataset=dataset, pad_idx=self.pad_idx, srtkey_fc=self.srtkey_fc, filter_fc=False, **kwargs)

""" Caution !!
This function is inperfect.
This will be a problem when you are doing non-text problem with n_inp > 1 (multiple input column),
which shouldn't be the case of huggingface/nlp user.
And I hope fastai come up with a good solution to show_batch multiple inputs problems for text/non-text.
"""
@typedispatch
def show_batch(x:tuple, y, samples, ctxs=None, max_n=9, **kwargs):
  if ctxs is None: ctxs = get_empty_df(min(len(samples), max_n))
  ctxs = show_batch[object](x, y, samples, max_n=max_n, ctxs=ctxs, **kwargs)
  display_df(pd.DataFrame(ctxs))
  return ctxs

class HF_Dataset():
  "A wrapper for `nlp.Dataset` to make it fulfill behaviors of `fastai2.Datasets` we need. Unless overidded, it behaves like original `nlp.Dataset`"
  
  def __init__(self, hf_dset, cols=None, hf_toker=None, neat_show=False, n_inp=1):
    """
    Args:
      `hf_dset` (`nlp.Dataset`)
      `cols` (`Optional`, default: `None`): **specify columns whose values form a output sample in order**, and the semantic type of each column to encode/decode, with one of the following signature.
      - `cols`(`Dict[Fastai Semantic Tensor]`): encode/decode {key} columns with {value} semantic tensor type. If {value} is `noop`, regard it as `TensorTuple` by default.
      - `cols`(`List[str]`): 
        - if of length 1, regard the 1st element as `TensorText`
        - if of length 2, regard the 1st element as `TensorText`, 2nd element as `TensorCategory`
        - Otherwise, regard all elements as `TensorTuple`
      - `None`: use `hf_dset.column_names` and deal with it like `List[str]` above.
      `hf_toker`: tokenizer of HuggingFace/Transformers
      `neat_show` (`Optional[bool]`, default:`False`): Show the original sentence instead of tokens joined.
      `n_inp (`int`, default:1) the first `n_inp` columns of `cols` is x, and the rest is y .
    """
    
    # some default setting for tensor type used in decoding
    if cols is None: cols = hf_dset.column_names
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
    store_attr(self, "hf_dset,cols,n_inp,hf_toker,neat_show")

  def __getitem__(self, idx):
    sample = self.hf_dset[idx]
    return tuple( tensor_cls(sample[col]) for col, tensor_cls in self.cols.items() )

  def __len__(self): return len(self.hf_dset)

  @property
  def col_names(self): return list(self.cols.keys())

  def decode(self, o, full=True): # `full` is for micmic `Dataset.decode` 
    if len(self.col_names) != len(o): return tuple( self._decode(o_) for o_ in o )
    return tuple( self._decode(o_, self.col_names[i]) for i, o_ in enumerate(o) )

  def _decode_title(self, d, title_cls, title=None):
    titled = title_cls(d)
    if title: titled._show_args['label'] = title
    return titled

  @typedispatch
  def _decode(self, t:torch.Tensor, title=None): return self._decode_title(t.tolist(), TitledTuple, title)

  @typedispatch
  def _decode(self, t:TensorText, title=None): 
    assert self.hf_toker, "You should give a huggingface tokenizer if you want to show batch."
    if self.neat_show: text = self.hf_toker.decode([idx for idx in t if idx != self.hf_toker.pad_token_id])
    else: text = ' '.join(self.hf_toker.convert_ids_to_tokens(t))
    return self._decode_title(text, TitledStr, title)

  @typedispatch
  def _decode(self, t:LMTensorText, title=None): return self._decode[TensorText](self, t, title)

  @typedispatch
  def _decode(self, t:TensorCategory, title=None): return self._decode_title(t.item(), Category, title)

  @typedispatch
  def _decode(self, t:TensorMultiCategory, title=None): return self._decode_title(t.tolist(), MultiCategory, title)

  def __getattr__(self, name):
    "If not defined, let the nlp.Dataset in it act for us."
    if name in HF_Dataset.__dict__: return HF_Dataset.__dict__[name]
    elif name in self.__dict__: return self.__dict__[name]
    elif hasattr(self.hf_dset, name): return getattr(self.hf_dset, name)
    raise AttributeError(f"Both 'HF_Dataset' object and 'nlp.Dataset' object have no '{name}' attribute ")
  
class HF_Datasets(FilteredBase):
  _dl_type,_dbunch_type = MySortedDL,DataLoaders
  
  @delegates(HF_Dataset.__init__)
  def __init__(self, hf_dsets: dict, test_with_label=False, **kwargs):
    """
    Args:
      `hf_dsets` (`Dict[nlp.Dataset]`): the order of dict items will be the order of `HF_Dataloader`s
      `test_with_label` (`bool`, default:`False`): whether testset come with labels.
    """
    cols, n_inp = kwargs.pop('cols', None), kwargs.get('n_inp', 1)
    self.hf_dsets = {};
    for split, dset in hf_dsets.items():
      if cols is None: cols = dset.column_names
      if split.startswith('test') and not test_with_label: 
        if isinstance(cols, list): _cols = cols[:n_inp]
        else: _cols = { k:v for _, (k,v) in zip(range(n_inp),cols.items()) }
      else: _cols = cols
      self.hf_dsets[split] = HF_Dataset(dset, cols=_cols, **kwargs)

  def subset(self, i): return list(self.hf_dsets.values())[i]
  def __getitem__(self, split): return self.hf_dsets[split]
  @property
  def n_subsets(self): return len(self.hf_dsets)
  @property
  def cache_dir(self): return Path(next(iter(self.hf_dsets.values())).cache_files[0]['filename']).parent
  
  @delegates(FilteredBase.dataloaders)
  def dataloaders(self, device='cpu', cache_dir=None, cache_name=None, **kwargs):
    """
    Args:
      `device` (`Optional[str]`, default:`'cpu'`)
      `cache_dir` (`Optional[str]`, default: `None`): directory to store dataloader caches. if `None`, use cache directory of first `nlp.Dataset` stored.
      `cache_name` (`Optional[str]`, default: `None`): format string with only one param `{split}` as cache file name under `cache_dir` for each split. If `None`, use autmatical cache path by hf/nlp.      
    """
    dl_kwargs = kwargs.pop('dl_kwargs', [{} for _ in range(len(self.hf_dsets))])
    # infer cache file names for each dataloader if needed
    dl_type = kwargs.pop('dl_type', self._dl_type)
    if dl_type==MySortedDL and cache_name:
      assert "{split}" in cache_name, "`cache_name` should be a string with '{split}' in it to be formatted."
      cache_dir = Path(cache_dir) if cache_dir else self.cache_dir
      cache_dir.mkdir(exist_ok=True)
      if not cache_name.endswith('.json'): cache_name += '.json'
      for i, split in enumerate(self.hf_dsets):
        dl_kwargs[i]['cache_file'] = cache_dir/cache_name.format(split=split)
    # change default to not drop last
    kwargs['drop_last'] = kwargs.pop('drop_last', False)
    # when corpus like glue/ax has only testset, set it to non-train setting
    if list(self.hf_dsets.keys())[0].startswith('test'):
      kwargs['shuffle_train'] = False
      kwargs['drop_last'] = False
    return super().dataloaders(dl_kwargs=dl_kwargs, device=device, **kwargs)

class HF_MergedDataset():
  def __init__(self, *datasets):
    self.dsets = datasets
    self.len = reduce(lambda a,d: a+len(d), self.dsets, 0)
  def __len__(self):
    return self.len
  def __getitem__(self, i):
    for dset in self.dsets:
      if i < len(dset): return dset[i]
      else: i -= len(dset)
    raise IndexError
  def set_format(self, type, columns):
    for dset in self.dsets: dset.set_format(type, columns)
  @property
  def format(self):
    form = self.dsets[0].format
    for dset in self.dsets:
      assert form == dset.format
    return form
  @property
  def cache_files(self):
    return concat(*[ds.cache_files for ds in self.dsets])

class HF_BaseTransform():
  "The base of HuggingFace/nlp transform"

  def __init__(self, hf_dsets, remove_original=False, out_cols=None):
    """
    Args:
      `hf_dsets` (`Dict[nlp.Dataset]` or `nlp.Dataset`): the dataset(s) to `map`
      `remove_original` (`bool`, default=`False`): whther to remove all original columns after `map`
      `out_cols` (`List[str]`, default=`None`): output column names. If specified, check they're not in the list of columns to remove when `remove_columns` is specified in `map` or `remove_original` is True
    """
    # check arguments
    if isinstance(hf_dsets, nlp.Dataset): hf_dsets = {'Single': hf_dsets}
    assert isinstance(hf_dsets, dict)
    # save attributes
    self.hf_dsets = hf_dsets
    self.remove_original,self.out_cols = remove_original,out_cols

  @property
  def cache_dir(self): return Path(next(iter(self.hf_dsets.values())).cache_files[0]['filename']).parent

  @delegates(nlp.Dataset.map, but=["cache_file_name"])
  def map(self, split_kwargs=None, cache_dir=None, cache_name=None, **kwargs):
    """
    Args:
      `split_kwargs` (`Dict[dict]` or `List[dict]`, default=`None`): arguments of `_map` and `nlp.Dataset.map` for specific splits. If specified in `dict`, you can specify only kwargs for some of all splits. 
      `cache_dir` (`Optional[str]`, default=`None`): if `None`, it is the cache dir of the (first) dataset. 
      `cache_name` (`Optional[str]`, default=`None`): format string with `{split}` converted to split name, as cache file name under `cache_dir` for each split. If `None`, use autmatical cache path by hf/nlp.  
    """
    # check/process arguments
    if self.remove_original: 
      assert 'remove_columns' not in kwargs, "You have specified to remove all original columns."
    if split_kwargs is None:
      split_kwargs = { split:{} for split in self.hf_dsets }
    elif isinstance(split_kwargs, list):
      split_kwargs = { split:split_kwargs[i] for i, split in enumerate(self.hf_dsets) }
    elif isinstance(split_kwargs, dict):
      for split in split_kwargs.keys(): assert split in self.hf_dsets, f"{split} is not the split names {list(self.hf_dsets.keys())}."
      for split in self.hf_dsets:
        if split not in split_kwargs: split_kwargs[split] = {}
    cache_dir = Path(cache_dir) if cache_dir else self.cache_dir
    cache_dir.mkdir(exist_ok=True)
    # map
    new_dsets = {}
    for split, dset in self.hf_dsets.items():
      if self.remove_original: kwargs['remove_columns'] = dset.column_names
      if cache_name: kwargs['cache_file_name'] = str(cache_dir/cache_name.format(split=split))
      kwargs.update(split_kwargs[split])
      if hasattr(kwargs, 'remove_columns'): self._check_outcols(kwargs['remove_columns'], split)
      new_dsets[split] = self._map(dset, split, **kwargs)
    # return
    if len(new_dsets)==1 and 'Single' in new_dsets: return new_dsets['Single']
    else: return new_dsets

  def _check_outcols(self, out_cols, rm_cols, split):
    if not self.out_cols: return
    for col in self.out_cols: assert col not in rm_cols, f"Output column name {col} is in the list of columns {rm_cols} will be removed after `map`." + f"The split is {split}" if split != 'Single' else ''

  # The default method you can override
  @delegates(nlp.Dataset.map)
  def _map(self, dset, split, **kwargs):
    return dset.map(self, **kwargs)
  
  # The method you need to implement
  def __call__(self, example): raise NotImplementedError

@delegates()
class HF_Transform(HF_BaseTransform):
  def __init__(self, hf_dset, func, **kwargs):
    """
    Args:
      `hf_dset` (`nlp.Dataset`),
      `func` (`Callable[dict]->dict`): sampel as `func` in `nlp.Dataset.map`
    """
    super().__init__(hf_dset, **kwargs)
    self.func = func
  def __call__(self, example): return self.func(example)

@delegates(but=["out_cols"])
class HF_TokenizeTfm(HF_BaseTransform):
  
  def __init__(self, hf_dset, cols, hf_toker, **kwargs):
    """
    Args:
      `hf_dset` (`nlp.Dataset`)
      `cols`: with one of the following signature:
        - `cols`(`Dict[str]`): tokenize the every column named key into column named its value
        - `cols`(`List[str]`): specify the name of columns to be tokenized, replace the original columns' data with tokenized one
      `hf_toker`: tokenizer of HuggingFace/Transformers.
    """
    if isinstance(cols, list): cols = {c:c for c in cols}
    assert isinstance(cols, dict)
    super().__init__(hf_dset, out_cols=list(cols.values()), **kwargs)
    self.cols, self.tokenizer = cols, hf_toker
    """
    If don't specify cache file name, it will be hashed binary of pickled function that
    passed to `map`, so if you pass the same function, it knows to use cache.
    But tokenizer can't be pickled, so use tokenizer config to make tfms use different 
    tokenizers unique.  
    """
    self.tokenizer_config = hf_toker.pretrained_init_configuration
  
  def __call__(self, example):
    for in_col, out_col in self.cols.items():
      example[out_col] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(example[in_col]))
    return example

  def __getstate__(self):
    "specify something you don't want pickle here, remember to use copy to not modfiy orginal instance"
    state = self.__dict__.copy() 
    state['tokenizer'] = None 
    return state

class AggregateTransform(HF_BaseTransform):
  """
  Inherit this class and implement `accumulate` and `create_example`
  """
  def __init__(self, hf_dset, inp_cols, out_cols, init_attrs, drop_last=False):
    """
    Args:
      `hf_dset` (`nlp.Dataset` or `Dict[nlp.Dataset]`)
      `inp_cols` (`List[str]`)
      `out_cols` (`List[str]`)
      `init_attrs` (`List[str]`): name of attributes of children class that need to be their initial status when starts to aggregate dataset. i.e. Those defined in `__init__` and the value will changed during `accumulate`
      `drop_last` (`Optional[bool]`, default: `False`): whether to drop the last accumulated sample.
      `cache_dir` (`str`, default=`None`): if `None`, it is the cache dir of the (first) dataset.
    """
    super().__init__(hf_dset)
    self.inp_cols, self.out_cols =  inp_cols, out_cols
    # batched map need dataset be in python format
    if isinstance(hf_dset, dict):
      for dset in hf_dset.values(): dset.set_format(type=None, columns=inp_cols) 
    else: hf_dset.set_format(type=None, columns=inp_cols)
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

  def _map(self, hf_dset, split, batch_size=1000, **kwargs):
    """
    Args:
      `batch_size`: see `nlp.Dataset.map`
    """
    assert 'remove_columns' not in kwargs, "Aggregation type transform will only leave output columns for output dataset."
    output_schema = self.get_output_schema(hf_dset, kwargs.pop('test_batch_size', 20))
    return hf_dset.map(function=self, batched=True, batch_size=batch_size, with_indices=True,
                            arrow_schema=output_schema, **kwargs)

  def get_output_schema(self, hf_dset, test_batch_size=20):
    "Do test run by ourself to get output schema, becuase default test run use batch_size=2, which might be too small to aggregate a sample out."
    """
    Args:
      `test_batch_size` (`int`, default=`20`): we infer the new schema of the aggregated dataset by the outputs of testing that passed first `test_batch_size` samples to aggregate. Depending how many sample aggreagted can you have a sample, this number might need to be higher.
    """
    test_inputs, test_indices = hf_dset[:test_batch_size], list(range(test_batch_size))
    test_output = self(test_inputs,test_indices)
    for col,val in test_output.items(): assert val, f"Didn't get any example in test, you might want to try larger `test_batch_size` than {test_batch_size}"
    assert sorted(self.out_cols) == sorted(test_output.keys()), f"Output columns are {self.out_cols}, but get example with {list(test_output.keys())}"
    return pa.Table.from_pydict(test_output).schema
    

@delegates(AggregateTransform, but=["inp_cols", "out_cols", "init_attrs"])
class LMTransform(AggregateTransform):
  def __init__(self, tokenized_hf_dset, max_len, text_col, x_text_col='x_text', y_text_col='y_text', **kwargs):
    if isinstance(text_col, str): text_col = {text_col:['x_text','y_text']}
    assert isinstance(text_col, dict)
    self.text_col, (self.x_text_col, self.y_text_col) = next(iter(text_col.items()))
    self._max_len = max_len + 1
    self.residual_len, self.new_text = self._max_len, []
    super().__init__(tokenized_hf_dset, inp_cols=[self.text_col], out_cols=[x_text_col, y_text_col], init_attrs=['residual_len', 'new_text'], **kwargs)
    

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

@delegates(AggregateTransform, but=["inp_cols", "out_cols", "init_attrs"])
class ELECTRADataTransform(AggregateTransform):
  
  def __init__(self, tokenized_hf_dset, text_col, max_length, cls_idx, sep_idx, **kwargs):
    if isinstance(text_col, str): text_col={text_col:text_col}
    assert isinstance(text_col, dict)
    self.in_col, self.out_col = next(iter(text_col.items()))
    self._current_sentences = []
    self._current_length = 0
    self._max_length = max_length
    self._target_length = max_length
    self.cls_idx, self.sep_idx = cls_idx, sep_idx
    super().__init__(tokenized_hf_dset, inp_cols=[self.in_col], out_cols=[self.out_col], 
                    init_attrs=['_current_sentences', '_current_length', '_target_length'], **kwargs)

  # two functions required by AggregateTransform
  def accumulate(self, tokids):
    self.add_line(tokids)
  
  def create_example(self):
    input_ids = self._create_example()
    return {self.out_col: input_ids}

  def add_line(self, tokids):
    """Adds a line of text to the current example being built."""
    self._current_sentences.append(tokids)
    self._current_length += len(tokids)
    if self._current_length >= self._target_length:
      self.commit_example(self.create_example())

  def _create_example(self):
    """Creates a pre-training example from the current list of sentences."""
    # small chance to only have one segment as in classification tasks
    if random.random() < 0.1:
      first_segment_target_length = 100000
    else:
      # -3 due to not yet having [CLS]/[SEP] tokens in the input text
      first_segment_target_length = (self._target_length - 3) // 2

    first_segment = []
    second_segment = []
    for sentence in self._current_sentences:
      # the sentence goes to the first segment if (1) the first segment is
      # empty, (2) the sentence doesn't put the first segment over length or
      # (3) 50% of the time when it does put the first segment over length
      if (len(first_segment) == 0 or
          len(first_segment) + len(sentence) < first_segment_target_length or
          (len(second_segment) == 0 and
           len(first_segment) < first_segment_target_length and
           random.random() < 0.5)):
        first_segment += sentence
      else:
        second_segment += sentence

    # trim to max_length while accounting for not-yet-added [CLS]/[SEP] tokens
    first_segment = first_segment[:self._max_length - 2]
    second_segment = second_segment[:max(0, self._max_length -
                                         len(first_segment) - 3)]

    # prepare to start building the next example
    self._current_sentences = []
    self._current_length = 0
    # small chance for random-length instead of max_length-length example
    if random.random() < 0.05:
      self._target_length = random.randint(5, self._max_length)
    else:
      self._target_length = self._max_length

    return self._make_example(first_segment, second_segment)

  def _make_example(self, first_segment, second_segment):
    """Converts two "segments" of text into a tf.train.Example."""
    input_ids = [self.cls_idx] + first_segment + [self.sep_idx]
    if second_segment:
      input_ids += second_segment + [self.sep_idx]
    return input_ids

# To just take hidden features output
class HF_Model(nn.Module):
  "A wrapper for model of HuggingFace/Transformers for using them with single input/output."
  def __init__(self, hf_cls, config_or_name, hf_toker=None, pad_id=None, sep_id=None, variable_sep=False):
    """
    Args:
      `hf_cls`: model class of HuggingFace/Transformers.
      `config_or_name`: a `config` instance or pretrained checkpoint name of HuggingFace/Transformers
      `hf_toker`: Used to infer `pad_id` and `spe_id`
      `pad_id`: To automatically infer attention mask. If not passed, use `hf_toker.pad_token_id`
      `sep_id`: To automatically infer token_type. If not passed, use `hf_toker.sep_token_id`
      `various_sep`: whether number of sep tokens in a sample could be 1 or 2.
    """
    "pass sep token id if sentence A sentence B setting. (default is sentence A setting)"
    super().__init__()
    if isinstance(config_or_name, str): self.model = hf_cls.from_pretrained(config_or_name)
    else: self.model = hf_cls(config_or_name)
    self.pad_id = pad_id if pad_id else hf_toker.pad_token_id
    self.sep_id = sep_id if sep_id else hf_toker.sep_token_id
    self.variable_sep = variable_sep
    
  def forward(self, x):
    attn_mask = x!= self.pad_id
    return self.model(x, attn_mask, token_type_ids=self._token_type_ids_for(x))[0]

  def _token_type_ids_for(self, x):
    "x: (batch size, sequence length)"
    if not self.variable_sep:
      num_sep = (x==self.sep_id).sum().item()
      if num_sep == x.shape[0]: return None
      assert num_sep == 2*x.shape[0], "Samples should all contains only one or all contains only two [SEP] in each of their texts"
    length = x.shape[1]
    tok_type_ids = []
    for s in x:
      second_sent_start = s.tolist().index(self.sep_id)+1
      tids = x.new_zeros(length).scatter(0,torch.arange(second_sent_start,length,device=x.device),1)
      tok_type_ids.append(tids)
    tok_type_ids = torch.stack(tok_type_ids)
    return tok_type_ids

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