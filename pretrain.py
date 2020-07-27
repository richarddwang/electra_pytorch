# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
from pathlib import Path
import os
from functools import partial
from datetime import datetime, timezone, timedelta
from IPython.core.debugger import set_trace as bk
import torch
from torch import nn
import torch.nn.functional as F
import torch.utils.data as D
import nlp
from transformers import ElectraModel, ElectraConfig, ElectraTokenizerFast, ElectraForMaskedLM,ElectraForPreTraining
from fastai2.text.all import *
import wandb
from fastai2.callback.wandb import WandbCallback
from _utils.would_like_to_pr import *
from _utils.huggingface import *
from _utils.utils import *


# %%
c = MyConfig({
    'device': 'cuda:1',
    'size': 'small',
    'use_fp16': True,
    'gen_smooth_label': False,
    'disc_smooth_label': False,
    'balanced_label': False,
    'tfdata': True,
    'sort_sample': True,
    'shuffle': True,
    'percent': False,
    # cache the data under it
    'cache_dir': Path.home()/"datasets", # ! should be `pathlib.Path` istance
    # cache checkpoints under checkpoint_dir/electra_pretrain
    'checkpoint_dir': Path.home()/'checkpoints', # ! should be `pathlib.Path` istance
})

i = ['small', 'base', 'large'].index(c.size)
c.mask_prob = [0.15, 0.15, 0.25][i]
c.lr = [5e-4, 2e-4, 2e-4][i]
c.bs = [128, 256, 2048][i]
c.steps = [10**6, 766*1000, 400*1000][i]
c.max_length = [128, 512, 512][i]
c.cache_dir.mkdir(exist_ok=True,parents=True)
c.checkpoint_dir.mkdir(exist_ok=True)
gen_config = ElectraConfig.from_pretrained(f'google/electra-{c.size}-generator')
disc_config = ElectraConfig.from_pretrained(f'google/electra-{c.size}-discriminator')
hf_tokenizer = ElectraTokenizerFast.from_pretrained(f"google/electra-{c.size}-generator")
if c.size in ['small', 'base']:
  wiki_cache_dir = c.cache_dir/"wikipedia/20200501.en/1.0.0"
  book_cache_dir = c.cache_dir/"bookcorpus/plain_text/1.0.0"
  wbdl_cache_dir = c.cache_dir/"wikibook_dl"
  wbdl_cache_dir.mkdir(exist_ok=True)
print(os.getpid())
print(c)

# %% [markdown]
# # 1. Load Data

# %%
if c.size in ['small', 'base'] and not c.tfdata:
  
  # wiki
  if (wiki_cache_dir/f"wiki_electra_{c.max_length}.arrow").exists():
    print('loading the electra data (wiki)')
    wiki = nlp.Dataset.from_file(str(wiki_cache_dir/f"wiki_electra_{c.max_length}.arrow"))
  else:
    print('load/download wiki dataset')
    wiki = nlp.load_dataset('wikipedia', '20200501.en', cache_dir=c.cache_dir)['train']
  
    print('creat data from wiki dataset for ELECTRA')
    wiki = ELECTRADataTransform(wiki, is_docs=True, text_col={'text':'input_ids'}, max_length=c.max_length, hf_toker=hf_tokenizer).map(cache_file_name=str(wiki_cache_dir/f"wiki_electra_{c.max_length}.arrow"))

  # bookcorpus
  if (book_cache_dir/f"book_electra_{c.max_length}.arrow").exists():
    print('loading the electra data (BookCorpus)')
    book = nlp.Dataset.from_file(str(book_cache_dir/f"book_electra_{c.max_length}.arrow"))
  else:
    print('load/download BookCorpus dataset')
    book = nlp.load_dataset('bookcorpus', cache_dir=c.cache_dir)['train']
  
    print('creat data from BookCorpus dataset for ELECTRA')
    book = ELECTRADataTransform(book, is_docs=False, text_col={'text':'input_ids'}, max_length=c.max_length, hf_toker=hf_tokenizer).map(cache_file_name=str(book_cache_dir/f"book_electra_{c.max_length}.arrow"))

  wb_data = HF_MergedDataset(wiki, book)
  wb_dsets = HF_Datasets({'train': wb_data}, cols=['input_ids'], hf_toker=hf_tokenizer)
  dls = wb_dsets.dataloaders(bs=c.bs, 
                             shuffle_train=c.shuffle,
                             srtkey_fc=None if c.sort_sample else False, 
                             cache_dir=Path.home()/'datasets/wikibook_dl', cache_name='dl_{split}.json')

else: # for large size
  #raise NotImplementedError
  import tensorflow as tf
  tf.compat.v1.enable_eager_execution()
  from electra.configure_pretraining import PretrainingConfig
  from electra.pretrain.pretrain_data import get_input_fn
  class TFDataLoader():
    def __init__(self, batch_size, size, device='cpu'):
      self.input_func = get_input_fn(PretrainingConfig('eeee', '../electra/data', **{'model_size': size}), True)
      self.bs = batch_size
      self.device = device
      self.n_inp = 1 # for fastai to split x and y
    def __iter__(self):
      infinite_data = self.input_func({'batch_size':self.bs})
      for features in infinite_data:
        yield (torch.tensor(features['input_ids'].numpy(), dtype=torch.long, device=self.device),)
    def to(self, device):
      self.device = device
      return self
    def __len__(self): return 62500 # 6.25% of 10**6
  dls = DataLoaders(TFDataLoader(c.bs, c.size))

# %% [markdown]
# # 2. Masked language model objective
# %% [markdown]
# ## 2.1 MLM objective callback

# %%
"""
Modified from HuggingFace/transformers (https://github.com/huggingface/transformers/blob/0a3d0e02c5af20bfe9091038c4fd11fb79175546/src/transformers/data/data_collator.py#L102). It is
- few ms faster: intead of a[b] a on gpu b on cpu, tensors here are all in the same device
- few tens of us faster: in how we create special token mask
- doesn't require huggingface tokenizer
- cost you only 20 ms on a (128,128) tensor, so dynamic masking is cheap   
"""
# https://github.com/huggingface/transformers/blob/1789c7daf1b8013006b0aef6cb1b8f80573031c5/examples/run_language_modeling.py#L179
def mask_tokens(inputs, mask_token_index, vocab_size, special_token_indices, mlm_probability=0.15, ignore_index=-100):
  """ Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original. """
  "ignore_index in nn.CrossEntropy is default to -100, so you don't need to specify ignore_index in loss"
  
  device = inputs.device
  labels = inputs.clone()
  # We sample a few tokens in each sequence for masked-LM training (with probability mlm_probability defaults to 0.15 in Bert/RoBER
  probability_matrix = torch.full(labels.shape, mlm_probability, device=device)
  special_tokens_mask = torch.full(inputs.shape, False, dtype=torch.bool, device=device)
  for sp_id in special_token_indices:
    special_tokens_mask = special_tokens_mask | (inputs==sp_id)
  probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
  mlm_mask = torch.bernoulli(probability_matrix).bool()
  labels[~mlm_mask] = ignore_index  # We only compute loss on masked tokens

  # 80% of the time, we replace masked input tokens with mask_token
  mask_token_mask = torch.bernoulli(torch.full(labels.shape, 0.8, device=device)).bool() & mlm_mask
  inputs[mask_token_mask] = mask_token_index

  # 10% of the time, we replace masked input tokens with random word
  replace_token_mask = torch.bernoulli(torch.full(labels.shape, 0.5, device=device)).bool() & mlm_mask & ~mask_token_mask
  random_words = torch.randint(vocab_size, labels.shape, dtype=torch.long, device=device)
  inputs[replace_token_mask] = random_words[replace_token_mask]

  # The rest of the time (10% of the time) we keep the masked input tokens unchanged
  return inputs, labels, mlm_mask

class MaskedLMCallback(Callback):
  @delegates(mask_tokens)
  def __init__(self, mask_tok_id, special_tok_ids, vocab_size, ignore_index=-100, for_electra=False, **kwargs):
    self.ignore_index = ignore_index
    self.for_electra = for_electra
    self.mask_tokens = partial(mask_tokens,
                               mask_token_index=mask_tok_id,
                               special_token_indices=special_tok_ids,
                               vocab_size=vocab_size,
                               ignore_index=-100,
                               **kwargs)

  def begin_batch(self):
    text_indices = self.xb[0]
    masked_inputs, labels, is_mlm_applied = self.mask_tokens(text_indices)
    if self.for_electra:
      self.learn.xb, self.learn.yb = (masked_inputs, is_mlm_applied, labels), (labels,)
    else:
      self.learn.xb, self.learn.yb = (masked_inputs,), (labels,)

  @delegates(TfmdDL.show_batch)
  def show_batch(self, dl, idx_show_ignored, verbose=True, **kwargs):
    b = dl.one_batch()
    inputs = b[0]
    masked_inputs, labels, is_mlm_applied = self.mask_tokens(inputs.clone())
    # check
    assert torch.equal(is_mlm_applied, labels!=self.ignore_index)
    assert torch.equal((~is_mlm_applied *masked_inputs + is_mlm_applied * labels), inputs)
    # change symbol to show the ignored position
    labels[labels==self.ignore_index] = idx_show_ignored
    # some notice to help understand the masking mechanism
    if verbose: 
      print("We won't count loss from position where y is ignore index")
      print("Notice 1. Positions have label token in y will be either [Mask]/other token/orginal token in x")
      print("Notice 2. Special tokens (CLS, SEP) won't be masked.")
      print("Notice 3. Dynamic masking: every time you run gives you different results.")
    # show
    tfm_b =(masked_inputs, is_mlm_applied, labels, labels) if self.for_electra else (masked_inputs, labels)   
    dl.show_batch(b=tfm_b, **kwargs)


# %%
mlm_cb = MaskedLMCallback(mask_tok_id=hf_tokenizer.mask_token_id, 
                          special_tok_ids=hf_tokenizer.all_special_ids, 
                          vocab_size=hf_tokenizer.vocab_size,
                          mlm_probability=c.mask_prob,
                          for_electra=True)

# %% [markdown]
# # 3. ELECTRA (replaced token detection objective)
# see details in paper [ELECTRA: Pre-training Text Encoders as Discriminators Rather Than Generators](https://arxiv.org/abs/2003.10555)

# %%
class ELECTRAModel(nn.Module):
  
  def __init__(self, generator, discriminator, pad_idx):
    super().__init__()
    self.generator, self.discriminator = generator,discriminator
    self.pad_idx = pad_idx
    self.gumbel_dist = torch.distributions.gumbel.Gumbel(0.,1.)
    self.toker = hf_tokenizer

  def to(self, *args, **kwargs):
    super().to(*args, **kwargs)
    a_tensor = next(self.parameters())
    device, dtype = a_tensor.device, a_tensor.dtype
    self.gumbel_dist = torch.distributions.gumbel.Gumbel(torch.tensor(0., device=device, dtype=dtype), torch.tensor(1., device=device, dtype=dtype))

  def forward(self, masked_inputs, is_mlm_applied, labels):
    # masked_inp_ids: (B,L)
    # ignored: (B,L)
    
    gen_logits = self.generator(masked_inputs) # (B, L, vocab size)

    # add gumbel noise and then sample
    pred_toks = (gen_logits + self.gumbel_dist.sample(gen_logits.shape)).argmax(dim=-1)
    # use predicted token to fill 15%(mlm prob) mlm applied positions
    generated = ~is_mlm_applied * masked_inputs + is_mlm_applied * pred_toks # (B,L)
    # not equal to generator predicted and is at mlm applied position
    is_replaced = (pred_toks != labels) * is_mlm_applied # (B, L)

    disc_logits = self.discriminator(generated) # (B, L)

    return gen_logits, generated, disc_logits, is_replaced

  def gumbel_softmax(self, logits):
    "reimplement it cuz there is a bug in torch.nn.functional.gumbel_softmax when fp16 (https://github.com/pytorch/pytorch/issues/41663)"
    "This is equal to the code of official ELECTRA repo. standard gumbel dist. = -ln(-ln(standard uniform dist.))"
    gumbel_noise = self.gumbel_dist.sample(logits.shape)
    return F.softmax(logits + gumbel_noise, dim=-1)

class ELECTRALoss():
  def __init__(self, pad_idx, loss_weights=(1.0, 50.0), gen_label_smooth=False, disc_label_smooth=False):
    self.pad_idx = pad_idx
    self.loss_weights = loss_weights
    if gen_label_smooth:
      eps = gen_label_smooth if isinstance(gen_label_smooth, float) else 0.1
      self.gen_loss_fc = LabelSmoothingCrossEntropyFlat(eps=eps)
    else:
      self.gen_loss_fc = CrossEntropyLossFlat()
    self.disc_loss_fc = nn.BCEWithLogitsLoss()
    self.disc_label_smooth = disc_label_smooth
    
  def __call__(self, pred, targ_ids):
    gen_logits, generated, disc_logits, is_replaced = pred
    gen_loss = self.gen_loss_fc(gen_logits.float(), targ_ids) # ignore position where targ_id==-100
    if c.balanced_label:
      non_mlm_pos = targ_ids == -100
      non_pad = generated != self.pad_idx
      rlm_mask = torch.full(non_pad.shape, c.mask_prob/(1-c.mask_prob), device=non_pad.device)
      rlm_mask = rlm_mask * non_pad * non_mlm_pos
      rlm_mask = (torch.bernoulli(rlm_mask).bool() + ~non_mlm_pos).bool()
      disc_logits = disc_logits.masked_select(rlm_mask) # -> 1d tensor
      is_replaced = is_replaced.masked_select(rlm_mask) # -> 1d tensor
    else:
      non_pad = generated != self.pad_idx
      disc_logits = disc_logits.masked_select(non_pad) # -> 1d tensor
      is_replaced = is_replaced.masked_select(non_pad) # -> 1d tensor
    if self.disc_label_smooth:
      eps = self.disc_label_smooth if isinstance(self.disc_label_smooth, float) else 0.1
      zeros = ~is_replaced
      is_replaced = is_replaced.float().masked_fill(zeros, eps)
    disc_loss = self.disc_loss_fc(disc_logits.float(), is_replaced.float())
    return gen_loss * self.loss_weights[0] + disc_loss * self.loss_weights[1]

  def decodes(self, pred):
    gen_logits, generated, disc_logits, is_replaced = pred
    gen_pred = gen_logits.argmax(dim=-1)
    disc_pred = disc_logits > 0
    return gen_pred, generated, disc_pred, is_replaced

# %% [markdown]
# # 4. Learning rate schedule

# %%
def linear_warmup_and_decay(pct_now, lr_max, end_lr, decay_power, total_steps,warmup_pct=None, warmup_steps=None):
  assert warmup_pct or warmup_steps
  if warmup_steps: warmup_pct = warmup_steps/total_steps
  """
  end_lr: the end learning rate for linear decay
  warmup_pct: percentage of training steps to for linear increase
  pct_now: percentage of traning steps we have gone through, notice pct_now=0.0 when calculating lr for first batch.
  """
  """
  pct updated after_batch, but global_step (in tf) seems to update before optimizer step,
  so pct is actually (global_step -1)/total_steps 
  """
  fixed_pct_now = pct_now + 1/total_steps
  """
  According to source code of the official repository, it seems they merged two lr schedule (warmup and linear decay)
  sequentially, instead of split training into two phases for each, this might because they think when in the early
  phase of training, pct is low, and thus the decaying formula makes little difference to lr.
  """
  decayed_lr = (lr_max-end_lr) * (1-fixed_pct_now)**decay_power + end_lr # https://www.tensorflow.org/api_docs/python/tf/compat/v1/train/polynomial_decay
  warmed_lr = decayed_lr * min(1.0, fixed_pct_now / warmup_pct) # https://github.com/google-research/electra/blob/81f7e5fc98b0ad8bfd20b641aa8bc9e6ac00c8eb/model/optimization.py#L44
  return warmed_lr


# %%
lr_shedule = ParamScheduler({'lr': partial(linear_warmup_and_decay,
                                            lr_max=c.lr,
                                            end_lr=0.0,
                                            decay_power=1,
                                            warmup_steps=10000 if not c.percent else int(0.1 * c.percent * 10**6),
                                            total_steps=c.steps if not c.percent else int(c.percent * c.steps))})

# %% [markdown]
# # 5. Train

# %%
def now_time():
  now_time = datetime.now(timezone(timedelta(hours=+8)))
  name = str(now_time)[6:-13].replace(' ', '_').replace(':', '-')
  return name


# %%
electra_model = ELECTRAModel(HF_Model(ElectraForMaskedLM, gen_config, hf_tokenizer, variable_sep=True), 
                             HF_Model(ElectraForPreTraining, disc_config, hf_tokenizer, variable_sep=True),
                             hf_tokenizer.pad_token_id,)
electra_loss_func = ELECTRALoss(pad_idx=hf_tokenizer.pad_token_id, gen_label_smooth=c.gen_smooth_label, disc_label_smooth=c.disc_smooth_label)

dls.to(torch.device(c.device))
run_name = now_time()
print(run_name)
learn = Learner(dls, electra_model,
                loss_func=electra_loss_func,
                opt_func=partial(Adam, eps=1e-6,),
                path=str(c.checkpoint_dir),
                model_dir='electra_pretrain',
                cbs=[mlm_cb,
                    RunSteps(c.steps, [0.0625, 0.125, 0.5, 1.0], run_name+"_{percent}") if not c.percent else RunSteps(int(c.steps*c.percent)),
                    ],
                )
if c.use_fp16: learn = learn.to_fp16()
learn.fit(9999, cbs=[lr_shedule])
if c.percent: learn.save(run_name)


