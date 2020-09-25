import random, re, os
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

def linear_warmup_and_decay(pct, lr_max, total_steps, warmup_steps=None, warmup_pct=None, end_lr=0.0, decay_power=1):
  """ pct (float): fastai count it as ith_step/num_epoch*len(dl), so we can't just use pct when our num_epoch is fake.he ith_step is count from 0, """
  if warmup_pct: warmup_steps = int(warmup_pct * total_steps)
  step_i = round(pct * total_steps)
  # According to the original source code, two schedules take effect at the same time, but decaying schedule will be neglible in the early time.
  decayed_lr = (lr_max-end_lr) * (1 - step_i/total_steps) ** decay_power + end_lr # https://www.tensorflow.org/api_docs/python/tf/compat/v1/train/polynomial_decay
  warmed_lr = decayed_lr * min(1.0, step_i/warmup_steps) # https://github.com/google-research/electra/blob/81f7e5fc98b0ad8bfd20b641aa8bc9e6ac00c8eb/model/optimization.py#L44
  return warmed_lr

def linear_warmup_and_then_decay(pct, lr_max, total_steps, warmup_steps=None, warmup_pct=None, end_lr=0.0, decay_power=1):
  """ pct (float): fastai count it as ith_step/num_epoch*len(dl), so we can't just use pct when our num_epoch is fake.he ith_step is count from 0, """
  if warmup_pct: warmup_steps = int(warmup_pct * total_steps)
  step_i = round(pct * total_steps)
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

class ConcatTransform(CombineTransform):
  def __init__(self, hf_dset, hf_tokenizer, max_length, text_col='text', book='multi'):
    super().__init__(hf_dset, in_cols=[text_col], out_cols=['input_ids', 'sentA_length'])
    self.max_length = max_length
    self.hf_tokenizer = hf_tokenizer
    self.book = book

  def reset_states(self):
    self.input_ids = [self.hf_tokenizer.cls_token_id]
    self.sent_lens = []

  def accumulate(self, sentence):
    if 'isbn' in sentence: return
    tokens = self.hf_tokenizer.convert_tokens_to_ids(self.hf_tokenizer.tokenize(sentence))
    tokens = tokens[:self.max_length-2] # trim sentence to max length if needed
    if self.book == 'single' or \
       (len(self.input_ids) + len(tokens) + 1 > self.max_length) or \
       (self.book == 'bi' and len(self.sent_lens)==2) :
      self.commit_example(self.create_example())
      self.reset_states()
    self.input_ids += [*tokens, self.hf_tokenizer.sep_token_id]
    self.sent_lens.append(len(tokens)+1)

  def create_example(self):
    if not self.sent_lens: return None
    self.sent_lens[0] += 1 # cls
    if self.book == 'multi':
      diff= 99999999
      for i in range(len(self.sent_lens)):
        current_diff = abs(sum(self.sent_lens[:i+1]) - sum(self.sent_lens[i+1:]))
        if current_diff > diff: break
        diff = current_diff
      return {'input_ids': self.input_ids, 'sentA_length': sum(self.sent_lens[:i])}
    else:
      return {'input_ids': self.input_ids, 'sentA_length': self.sent_lens[0]}

class ELECTRADataProcessor(object):
  """Given a stream of input text, creates pretraining examples."""

  def __init__(self, hf_dset, hf_tokenizer, max_length, text_col='text', lines_delimiter='\n', minimize_data_size=True, apply_cleaning=True):
    self.hf_tokenizer = hf_tokenizer
    self._current_sentences = []
    self._current_length = 0
    self._max_length = max_length
    self._target_length = max_length

    self.hf_dset = hf_dset
    self.text_col = text_col
    self.lines_delimiter = lines_delimiter
    self.minimize_data_size = minimize_data_size
    self.apply_cleaning = apply_cleaning

  def map(self, **kwargs):
    "Some settings of datasets.Dataset.map for ELECTRA data processing"
    num_proc = kwargs.pop('num_proc', os.cpu_count())
    return self.hf_dset.my_map(
      function=self,
      batched=True,
      remove_columns=self.hf_dset.column_names, # this is must b/c we will return different number of rows
      disable_nullable=True,
      input_columns=[self.text_col],
      writer_batch_size=10**4,
      num_proc=num_proc,
      **kwargs
    )

  def __call__(self, texts):
    if self.minimize_data_size: new_example = {'input_ids':[], 'sentA_length':[]}
    else: new_example = {'input_ids':[], 'input_mask': [], 'segment_ids': []}

    for text in texts: # for every doc
      
      for line in re.split(self.lines_delimiter, text): # for every paragraph
        
        if re.fullmatch(r'\s*', line): continue # empty string or string with all space characters
        if self.apply_cleaning and self.filter_out(line): continue
        
        example = self.add_line(line)
        if example:
          for k,v in example.items(): new_example[k].append(v)
      
      if self._current_length != 0:
        example = self._create_example()
        for k,v in example.items(): new_example[k].append(v)

    return new_example

  def filter_out(self, line):
    if len(line) < 80: return True
    return False 

  def clean(self, line):
    # () is remainder after link in it filtered out
    return line.strip().replace("\n", " ").replace("()","")

  def add_line(self, line):
    """Adds a line of text to the current example being built."""
    line = self.clean(line)
    tokens = self.hf_tokenizer.tokenize(line)
    tokids = self.hf_tokenizer.convert_tokens_to_ids(tokens)
    self._current_sentences.append(tokids)
    self._current_length += len(tokids)
    if self._current_length >= self._target_length:
      return self._create_example()
    return None

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
    input_ids = [self.hf_tokenizer.cls_token_id] + first_segment + [self.hf_tokenizer.sep_token_id]
    sentA_length = len(input_ids)
    segment_ids = [0] * sentA_length
    if second_segment:
      input_ids += second_segment + [self.hf_tokenizer.sep_token_id]
      segment_ids += [1] * (len(second_segment) + 1)
    
    if self.minimize_data_size:
      return {
        'input_ids': input_ids,
        'sentA_length': sentA_length,
      }
    else:
      input_mask = [1] * len(input_ids)
      input_ids += [0] * (self._max_length - len(input_ids))
      input_mask += [0] * (self._max_length - len(input_mask))
      segment_ids += [0] * (self._max_length - len(segment_ids))
      return {
        'input_ids': input_ids,
        'input_mask': input_mask,
        'segment_ids': segment_ids,
      }