from functools import partial
from fastai2.text.all import *
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
  return warmed_lr

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

@delegates(CombineTransform, but=["inp_cols", "out_cols", "init_attrs"])
class ELECTRADataTransform(CombineTransform):
  "Process any text corpus for ELECTRA's use"
  def __init__(self, hf_dset, is_docs, text_col, max_length, hf_toker, delimiter='\n', **kwargs):
    """
    Args:
      hf_dset (:class:`nlp.Dataset` or Dict[:class:`nlp.Dataset`]): **untokenized** Hugging Face dataset(s) to do the transform
      is_docs (bool): Whether each sample of this dataset is a doc
      text_col (str): the name of column of the dataset contains text 
      max_length (str): max length of each sentence
      hf_toker (:class:`transformers.PreTrainedTokenizer`): Hugging Face tokenizer
      delimiter (str): what is the delimiter to segment sentences in the input text
      kwargs: passed to :class:`CombineTransform`
    """
    self.is_docs = is_docs
    self.in_col = text_col
    self._current_sentences = []
    self._current_length = 0
    self._max_length = max_length
    self._target_length = max_length
    self.cls_idx, self.sep_idx = hf_toker.cls_token_id, hf_toker.sep_token_id
    self.hf_toker = hf_toker
    self.delimiter = delimiter
    super().__init__(hf_dset, inp_cols=[self.in_col], out_cols=['input_ids','sentA_lenth'], 
                    init_attrs=['_current_sentences', '_current_length', '_target_length'], **kwargs)

  """
  This two main functions adapts official source code creates pretraining dataset, to CombineTransform
  """
  def accumulate(self, text):
    sentences = text.split(self.delimiter)
    for sentence in sentences:
      if not sentence: continue # skip empty
      tokids = self.hf_toker.convert_tokens_to_ids(self.hf_toker.tokenize(sentence))
      self.add_line(tokids)
    # end of doc
    if self.is_docs and self._current_length > 0:
      self.commit_example(self.create_example())
  
  def create_example(self):
    input_ids, sentA_lenth = self._create_example() # this line reset _current_sentences and _current_length in the end
    return {'input_ids': input_ids, 'sentA_lenth':sentA_lenth}
  # ...................................................

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
    sentA_lenth = len(input_ids)
    if second_segment:
      input_ids += second_segment + [self.sep_idx]
    return input_ids, sentA_lenth

  def __getstate__(self):
    "specify something you don't want pickle here, remember to use copy to not modfiy orginal instance"
    state = self.__dict__.copy() 
    state['hf_toker'] = None 
    return state