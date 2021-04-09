# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import os, sys, random
from pathlib import Path
from functools import partial
from datetime import datetime, timezone, timedelta
from IPython.core.debugger import set_trace as bk
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import torch.tensor as T
import datasets
from fastai.text.all import *
from transformers import ElectraConfig, ElectraTokenizerFast, ElectraForMaskedLM, ElectraForPreTraining
from hugdatafast import *
from _utils.utils import *
from _utils.would_like_to_pr import *

# %% [markdown]
# # 1. Configuraton

# %%
c = MyConfig({
    'device': 'cuda:0',
    
    'base_run_name': 'vanilla', # run_name = {base_run_name}_{seed}
    'seed': 11081, # 11081 36 1188 76 1 4 4649 7 # None/False to randomly choose seed from [0,999999]

    'adam_bias_correction': False,
    'schedule': 'original_linear',
    'sampling': 'fp32_gumbel',
    'electra_mask_style': True,
    'gen_smooth_label': False,
    'disc_smooth_label': False,

    'size': 'small',
    'datas': ['openwebtext'],
    
    'logger': 'wandb',
    'num_workers': 3,
    'my_model': False, # only for my personal research
})

# only for my personal research
hparam_update = {
    
}

""" Vanilla ELECTRA settings
'adam_bias_correction': False,
'schedule': 'original_linear',
'sampling': 'fp32_gumbel',
'electra_mask_style': True,
'gen_smooth_label': False,
'disc_smooth_label': False,
'size': 'small',
'datas': ['openwebtext'],
"""


# %%
# Check and Default
assert c.sampling in ['fp32_gumbel', 'fp16_gumbel', 'multinomial']
assert c.schedule in ['original_linear', 'separate_linear', 'one_cycle', 'adjusted_one_cycle']
for data in c.datas: assert data in ['wikipedia', 'bookcorpus', 'openwebtext']
assert c.logger in ['wandb', 'neptune', None, False]
if not c.base_run_name: c.base_run_name = str(datetime.now(timezone(timedelta(hours=+8))))[6:-13].replace(' ','').replace(':','').replace('-','')
if not c.seed: c.seed = random.randint(0, 999999)
c.run_name = f'{c.base_run_name}_{c.seed}'
if c.gen_smooth_label is True: c.gen_smooth_label = 0.1
if c.disc_smooth_label is True: c.disc_smooth_label = 0.1

# Setting of different sizes
i = ['small', 'base', 'large'].index(c.size)
c.mask_prob = [0.15, 0.15, 0.25][i]
c.lr = [5e-4, 2e-4, 2e-4][i]
c.bs = [128, 256, 2048][i]
c.steps = [10**6, 766*1000, 400*1000][i]
c.max_length = [128, 512, 512][i]
generator_size_divisor = [4, 3, 4][i]
disc_config = ElectraConfig.from_pretrained(f'google/electra-{c.size}-discriminator')
gen_config = ElectraConfig.from_pretrained(f'google/electra-{c.size}-generator')
# note that public electra-small model is actually small++ and don't scale down generator size 
gen_config.hidden_size = int(disc_config.hidden_size/generator_size_divisor)
gen_config.num_attention_heads = disc_config.num_attention_heads//generator_size_divisor
gen_config.intermediate_size = disc_config.intermediate_size//generator_size_divisor
hf_tokenizer = ElectraTokenizerFast.from_pretrained(f"google/electra-{c.size}-generator")

# logger
if c.logger == 'neptune':
  import neptune
  from fastai.callback.neptune import NeptuneCallback
  neptune.init(project_qualified_name='richard-wang/electra-pretrain')
elif c.logger == 'wandb':
  import wandb
  from fastai.callback.wandb import  WandbCallback

# Path to data
Path('./datasets', exist_ok=True)
Path('./checkpoints/pretrain').mkdir(exist_ok=True, parents=True)
edl_cache_dir = Path("./datasets/electra_dataloader")
edl_cache_dir.mkdir(exist_ok=True)

# Print info
print(f"process id: {os.getpid()}")
print(c)
print(hparam_update)


# %%
if c.my_model: # only for use of my personal research 
  sys.path.insert(0, os.path.abspath(".."))
  from modeling.model import ModelForGenerator,ModelForDiscriminator
  from hyperparameter import electra_hparam_from_hf
  gen_hparam = electra_hparam_from_hf(gen_config, hf_tokenizer)
  gen_hparam.update(hparam_update)
  disc_hparam = electra_hparam_from_hf(disc_config, hf_tokenizer)
  disc_hparam.update(hparam_update)

# %% [markdown]
# # 1. Load Data

# %%
dsets = []
ELECTRAProcessor = partial(ELECTRADataProcessor, hf_tokenizer=hf_tokenizer, max_length=c.max_length)

# Wikipedia
if 'wikipedia' in c.datas:
  print('load/download wiki dataset')
  wiki = datasets.load_dataset('wikipedia', '20200501.en', cache_dir='./datasets')['train']
  print('load/create data from wiki dataset for ELECTRA')
  e_wiki = ELECTRAProcessor(wiki).map(cache_file_name=f"electra_wiki_{c.max_length}.arrow", num_proc=1)
  dsets.append(e_wiki)

# OpenWebText
if 'openwebtext' in c.datas:
  print('load/download OpenWebText Corpus')
  owt = datasets.load_dataset('openwebtext', cache_dir='./datasets')['train']
  print('load/create data from OpenWebText Corpus for ELECTRA')
  e_owt = ELECTRAProcessor(owt, apply_cleaning=False).map(cache_file_name=f"electra_owt_{c.max_length}.arrow", num_proc=1)
  dsets.append(e_owt)

assert len(dsets) == len(c.datas)

merged_dsets = {'train': datasets.concatenate_datasets(dsets)}
hf_dsets = HF_Datasets(merged_dsets, cols={'input_ids':TensorText,'sentA_length':noop},
                       hf_toker=hf_tokenizer, n_inp=2)
dls = hf_dsets.dataloaders(bs=c.bs, num_workers=c.num_workers, pin_memory=False,
                           shuffle_train=True,
                           srtkey_fc=False, 
                           cache_dir='./datasets/electra_dataloader', cache_name='dl_{split}.json')

# %% [markdown]
# # 2. Masked language model objective
# %% [markdown]
# ## 2.1 MLM objective callback

# %%
"""
Modified from HuggingFace/transformers (https://github.com/huggingface/transformers/blob/0a3d0e02c5af20bfe9091038c4fd11fb79175546/src/transformers/data/data_collator.py#L102). 
It is a little bit faster cuz 
- intead of a[b] a on gpu b on cpu, tensors here are all in the same device
- don't iterate the tensor when create special tokens mask
And
- doesn't require huggingface tokenizer
- cost you only 550 µs for a (128,128) tensor on gpu, so dynamic masking is cheap   
"""
def mask_tokens(inputs, mask_token_index, vocab_size, special_token_indices, mlm_probability=0.15, replace_prob=0.1, orginal_prob=0.1, ignore_index=-100):
  """ 
  Prepare masked tokens inputs/labels for masked language modeling: (1-replace_prob-orginal_prob)% MASK, replace_prob% random, orginal_prob% original within mlm_probability% of tokens in the sentence. 
  * ignore_index in nn.CrossEntropy is default to -100, so you don't need to specify ignore_index in loss
  """
  
  device = inputs.device
  labels = inputs.clone()
  
  # Get positions to apply mlm (mask/replace/not changed). (mlm_probability)
  probability_matrix = torch.full(labels.shape, mlm_probability, device=device)
  special_tokens_mask = torch.full(inputs.shape, False, dtype=torch.bool, device=device)
  for sp_id in special_token_indices:
    special_tokens_mask = special_tokens_mask | (inputs==sp_id)
  probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
  mlm_mask = torch.bernoulli(probability_matrix).bool()
  labels[~mlm_mask] = ignore_index  # We only compute loss on mlm applied tokens

  # mask  (mlm_probability * (1-replace_prob-orginal_prob))
  mask_prob = 1 - replace_prob - orginal_prob
  mask_token_mask = torch.bernoulli(torch.full(labels.shape, mask_prob, device=device)).bool() & mlm_mask
  inputs[mask_token_mask] = mask_token_index

  # replace with a random token (mlm_probability * replace_prob)
  if int(replace_prob)!=0:
    rep_prob = replace_prob/(replace_prob + orginal_prob)
    replace_token_mask = torch.bernoulli(torch.full(labels.shape, rep_prob, device=device)).bool() & mlm_mask & ~mask_token_mask
    random_words = torch.randint(vocab_size, labels.shape, dtype=torch.long, device=device)
    inputs[replace_token_mask] = random_words[replace_token_mask]

  # do nothing (mlm_probability * orginal_prob)
  pass

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

  def before_batch(self):
    input_ids, sentA_lenths  = self.xb
    masked_inputs, labels, is_mlm_applied = self.mask_tokens(input_ids)
    if self.for_electra:
      self.learn.xb, self.learn.yb = (masked_inputs, sentA_lenths, is_mlm_applied, labels), (labels,)
    else:
      self.learn.xb, self.learn.yb = (masked_inputs, sentA_lenths), (labels,)

  @delegates(TfmdDL.show_batch)
  def show_batch(self, dl, idx_show_ignored, verbose=True, **kwargs):
    b = dl.one_batch()
    input_ids, sentA_lenths  = b
    masked_inputs, labels, is_mlm_applied = self.mask_tokens(input_ids.clone())
    # check
    assert torch.equal(is_mlm_applied, labels!=self.ignore_index)
    assert torch.equal((~is_mlm_applied *masked_inputs + is_mlm_applied * labels), input_ids)
    # change symbol to show the ignored position
    labels[labels==self.ignore_index] = idx_show_ignored
    # some notice to help understand the masking mechanism
    if verbose: 
      print("We won't count loss from position where y is ignore index")
      print("Notice 1. Positions have label token in y will be either [Mask]/other token/orginal token in x")
      print("Notice 2. Special tokens (CLS, SEP) won't be masked.")
      print("Notice 3. Dynamic masking: every time you run gives you different results.")
    # show
    tfm_b =(masked_inputs, sentA_lenths, is_mlm_applied, labels) if self.for_electra else (masked_inputs, sentA_lenths, labels)   
    dl.show_batch(b=tfm_b, **kwargs)


# %%
mlm_cb = MaskedLMCallback(mask_tok_id=hf_tokenizer.mask_token_id, 
                          special_tok_ids=hf_tokenizer.all_special_ids, 
                          vocab_size=hf_tokenizer.vocab_size,
                          mlm_probability=c.mask_prob,
                          replace_prob=0.0 if c.electra_mask_style else 0.1, 
                          orginal_prob=0.15 if c.electra_mask_style else 0.1,
                          for_electra=True)
#mlm_cb.show_batch(dls[0], idx_show_ignored=hf_tokenizer.convert_tokens_to_ids(['#'])[0])

# %% [markdown]
# # 3. ELECTRA (replaced token detection objective)
# see details in paper [ELECTRA: Pre-training Text Encoders as Discriminators Rather Than Generators](https://arxiv.org/abs/2003.10555)

# %%
class ELECTRAModel(nn.Module):
  
  def __init__(self, generator, discriminator, hf_tokenizer):
    super().__init__()
    self.generator, self.discriminator = generator,discriminator
    self.gumbel_dist = torch.distributions.gumbel.Gumbel(0.,1.)
    self.hf_tokenizer = hf_tokenizer

  def to(self, *args, **kwargs):
    "Also set dtype and device of contained gumbel distribution if needed"
    super().to(*args, **kwargs)
    a_tensor = next(self.parameters())
    device, dtype = a_tensor.device, a_tensor.dtype
    if c.sampling=='fp32_gumbel': dtype = torch.float32
    self.gumbel_dist = torch.distributions.gumbel.Gumbel(torch.tensor(0., device=device, dtype=dtype), torch.tensor(1., device=device, dtype=dtype))

  def forward(self, masked_inputs, sentA_lenths, is_mlm_applied, labels):
    """
    masked_inputs (Tensor[int]): (B, L)
    sentA_lenths (Tensor[int]): (B, L)
    is_mlm_applied (Tensor[boolean]): (B, L), True for positions chosen by mlm probability 
    labels (Tensor[int]): (B, L), -100 for positions where are not mlm applied
    """
    attention_mask, token_type_ids = self._get_pad_mask_and_token_type(masked_inputs, sentA_lenths)
    if c.my_model:
      gen_logits = self.generator(masked_inputs, attention_mask, token_type_ids, is_mlm_applied)[0]
      # already reduced before the mlm output layer, save more space and speed
      mlm_gen_logits = gen_logits # ( #mlm_positions, vocab_size)
    else:
      gen_logits = self.generator(masked_inputs, attention_mask, token_type_ids)[0] # (B, L, vocab size)
      # reduce size to save space and speed
      mlm_gen_logits = gen_logits[is_mlm_applied, :] # ( #mlm_positions, vocab_size)
    
    with torch.no_grad():
      # sampling
      pred_toks = self.sample(mlm_gen_logits) # ( #mlm_positions, )
      # produce inputs for discriminator
      generated = masked_inputs.clone() # (B,L)
      generated[is_mlm_applied] = pred_toks # (B,L)
      # produce labels for discriminator
      is_replaced = is_mlm_applied.clone() # (B,L)
      is_replaced[is_mlm_applied] = (pred_toks != labels[is_mlm_applied]) # (B,L)

    disc_logits = self.discriminator(generated, attention_mask, token_type_ids)[0] # (B, L)

    return mlm_gen_logits, generated, disc_logits, is_replaced, attention_mask, is_mlm_applied

  def _get_pad_mask_and_token_type(self, input_ids, sentA_lenths):
    """
    Only cost you about 500 µs for (128, 128) on GPU, but so that your dataset won't need to save attention_mask and token_type_ids and won't be unnecessarily large, thus, prevent cpu processes loading batches from consuming lots of cpu memory and slow down the machine. 
    """
    attention_mask = input_ids != self.hf_tokenizer.pad_token_id
    seq_len = input_ids.shape[1]
    token_type_ids = torch.tensor([ ([0]*len + [1]*(seq_len-len)) for len in sentA_lenths.tolist()],  
                                  device=input_ids.device)
    return attention_mask, token_type_ids

  def sample(self, logits):
    "Reimplement gumbel softmax cuz there is a bug in torch.nn.functional.gumbel_softmax when fp16 (https://github.com/pytorch/pytorch/issues/41663). Gumbel softmax is equal to what official ELECTRA code do, standard gumbel dist. = -ln(-ln(standard uniform dist.))"
    if c.sampling == 'fp32_gumbel':
      gumbel = self.gumbel_dist.sample(logits.shape).to(logits.device)
      return (logits.float() + gumbel).argmax(dim=-1)
    elif c.sampling == 'fp16_gumbel':  # 5.06 ms
      gumbel = self.gumbel_dist.sample(logits.shape).to(logits.device)
      return (logits + gumbel).argmax(dim=-1)
    elif c.sampling == 'multinomial':  # 2.X ms
      return torch.multinomial(F.softmax(logits, dim=-1), 1).squeeze()

class ELECTRALoss():
  def __init__(self, loss_weights=(1.0, 50.0), gen_label_smooth=False, disc_label_smooth=False):
    self.loss_weights = loss_weights
    self.gen_loss_fc = LabelSmoothingCrossEntropyFlat(eps=gen_label_smooth) if gen_label_smooth else CrossEntropyLossFlat()
    self.disc_loss_fc = nn.BCEWithLogitsLoss()
    self.disc_label_smooth = disc_label_smooth
    
  def __call__(self, pred, targ_ids):
    mlm_gen_logits, generated, disc_logits, is_replaced, non_pad, is_mlm_applied = pred
    gen_loss = self.gen_loss_fc(mlm_gen_logits.float(), targ_ids[is_mlm_applied])
    disc_logits = disc_logits.masked_select(non_pad) # -> 1d tensor
    is_replaced = is_replaced.masked_select(non_pad) # -> 1d tensor
    if self.disc_label_smooth:
      is_replaced = is_replaced.float().masked_fill(~is_replaced, self.disc_label_smooth)
    disc_loss = self.disc_loss_fc(disc_logits.float(), is_replaced.float())
    return gen_loss * self.loss_weights[0] + disc_loss * self.loss_weights[1]

# %% [markdown]
# # 5. Train

# %%
# Seed & PyTorch benchmark
torch.backends.cudnn.benchmark = True
dls[0].rng = random.Random(c.seed) # for fastai dataloader
random.seed(c.seed)
np.random.seed(c.seed)
torch.manual_seed(c.seed)

# Generator and Discriminator
if c.my_model:
  generator = ModelForGenerator(gen_hparam)
  discriminator = ModelForDiscriminator(disc_hparam)
  discriminator.electra.embedding = generator.electra.embedding
  # implicitly tie in/out embeddings of generator
else:
  generator = ElectraForMaskedLM(gen_config)
  discriminator = ElectraForPreTraining(disc_config)
  discriminator.electra.embeddings = generator.electra.embeddings
  generator.generator_lm_head.weight = generator.electra.embeddings.word_embeddings.weight

# ELECTRA training loop
electra_model = ELECTRAModel(generator, discriminator, hf_tokenizer)
electra_loss_func = ELECTRALoss(gen_label_smooth=c.gen_smooth_label, disc_label_smooth=c.disc_smooth_label)

# Optimizer
if c.adam_bias_correction: opt_func = partial(Adam, eps=1e-6, mom=0.9, sqr_mom=0.999, wd=0.01)
else: opt_func = partial(Adam_no_bias_correction, eps=1e-6, mom=0.9, sqr_mom=0.999, wd=0.01)

# Learning rate shedule
if c.schedule.endswith('linear'):
  lr_shed_func = linear_warmup_and_then_decay if c.schedule=='separate_linear' else linear_warmup_and_decay
  lr_shedule = ParamScheduler({'lr': partial(lr_shed_func,
                                             lr_max=c.lr,
                                             warmup_steps=10000,
                                             total_steps=c.steps,)})


# Learner
dls.to(torch.device(c.device))
learn = Learner(dls, electra_model,
                loss_func=electra_loss_func,
                opt_func=opt_func ,
                path='./checkpoints',
                model_dir='pretrain',
                cbs=[mlm_cb,
                    RunSteps(c.steps, [0.0625, 0.125, 0.25, 0.5, 1.0], c.run_name+"_{percent}"),
                     ],
                )

# logging
if c.logger == 'neptune':
  neptune.create_experiment(name=c.run_name, params={**c, **hparam_update})
  learn.add_cb(NeptuneCallback(log_model_weights=False))
elif c.logger == 'wandb':
  wandb.init(name=c.run_name, project='electra_pretrain', config={**c, **hparam_update})
  learn.add_cb(WandbCallback(log_preds=False, log_model=False))

# Mixed precison and Gradient clip
learn.to_native_fp16(init_scale=2.**11)
learn.add_cb(GradientClipping(1.))

# Print time and run name
print(f"{c.run_name} , starts at {datetime.now()}")

# Run
if c.schedule == 'one_cycle': learn.fit_one_cycle(9999, lr_max=c.lr)
elif c.schedule == 'adjusted_one_cycle': learn.fit_one_cycle(9999, lr_max=c.lr, div=1e5, pct_start=10000/c.steps)
else: learn.fit(9999, cbs=[lr_shedule])


