# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
from pathlib import Path
from functools import partial
from itertools import product
from IPython.core.debugger import set_trace as bk
import pandas as pd
import torch
from torch import nn
import nlp
from transformers import ElectraModel, ElectraConfig, ElectraTokenizerFast, ElectraForPreTraining
from fastai2.text.all import *
import wandb
from fastai2.callback.wandb import WandbCallback
from _utils.would_like_to_pr import *
from _utils.huggingface import *
from _utils.wsc import *


# %%
SIZE = 'small'
assert SIZE in ['small', 'base', 'large']
hf_tokenizer = ElectraTokenizerFast.from_pretrained(f"google/electra-{SIZE}-discriminator")
electra_config = ElectraConfig.from_pretrained(f'google/electra-{SIZE}-discriminator')
CONFIG = {
  'lr': [3e-4, 1e-4, 5e-5],
  'layer_lr_decay': [0.8,0.8,0.9],
}
I = ['small', 'base', 'large'].index(SIZE)
config = {k:vals[I] for k,vals in CONFIG.items()}

config.update({
  'max_length': 512,
  'use_wsc': False,
  'use_fp16': False,
})

# %% [markdown]
# # 1. Prepare data

# %%
cache_dir = Path.home()/'datasets'
cache_dir.mkdir(parents=True, exist_ok=True)

# %% [markdown]
# ## 1.1 Download and Preprocess

# %%
def textcols(task):
  "Infer text cols of different GLUE datasets in huggingface/nlp"
  if task in ['qnli']: return ['question', 'sentence']
  elif task in ['mrpc','stsb','wnli','rte']: return ['sentence1', 'sentence2']
  elif task in ['qqp']: return ['question1','question2']
  elif task in ['mnli','ax']: return ['premise','hypothesis']
  elif task in ['cola','sst2']: return ['sentence']

def tokenize_sents(example, cols):
  example['inp_ids'] = hf_tokenizer.encode(*[ example[c] for c in cols])
  return example

def tokenize_sents_max_len(example, cols, max_length):
  # Follow BERT and ELECTRA, we truncate examples longer than max length, see https://github.com/google-research/electra/blob/79111328070e491b287c307906701ebc61091eb2/finetune/classification/classification_tasks.py#L296
  tokens_a = hf_tokenizer.tokenize(example[cols[0]])
  tokens_b = hf_tokenizer.tokenize(example[cols[1]]) if len(cols)==2 else []
  _max_length = max_length - 1 - len(cols) # preserved for cls and sep tokens
  while True:
    total_length = len(tokens_a) + len(tokens_b)
    if total_length <= _max_length:
      break
    if len(tokens_a) > len(tokens_b):
      tokens_a.pop()
    else:
      tokens_b.pop()
  tokens = [hf_tokenizer.cls_token, *tokens_a, hf_tokenizer.sep_token]
  if tokens_b: tokens += [*tokens_b, hf_tokenizer.sep_token]
  example['inp_ids'] = hf_tokenizer.convert_tokens_to_ids(tokens)
  return example

# get tokenized datasets and dataloaders
glue_dsets = {}; glue_dls = {}
for task in ['cola', 'sst2', 'mrpc', 'qqp', 'stsb', 'mnli', 'qnli', 'rte', 'wnli', 'ax']:
  # General case and special case for WSC
  if task == 'wnli' and config['use_wsc']:
    benchmark, subtask = 'super_glue', 'wsc.fixed'
    # samples in all splits are all less than 128-2, so don't need to worry about max_length
    Tfm = partial(WSCTransform, hf_toker=hf_tokenizer)
    cols = {'inp_ids':TensorText, 'span': noop, 'label':TensorCategory}
    n_inp=2
    cache_name = "tokenized_{split}.arrow"
  else:
    benchmark, subtask = 'glue', task
    tok_func = partial(tokenize_sents_max_len, cols=textcols(task), max_length=config['max_length'])
    Tfm = partial(HF_Transform, func=tok_func)
    cols = ['inp_ids', 'label']
    n_inp=1
    cache_name = f"tokenized_{config['max_length']}_{{split}}.arrow"
  # load / download datasets.
  dsets = nlp.load_dataset(benchmark, subtask, cache_dir=cache_dir)
  # There is two samples broken in QQP training set
  if task=='qqp': dsets['train'] = dsets['train'].filter(lambda e: e['question2']!='',
                                          cache_file_name=str(cache_dir/'glue/qqp/1.0.0/fixed_train.arrow'))
  # load / make tokenized datasets
  glue_dsets[task] = Tfm(dsets).map(cache_name=cache_name)
  # load / make dataloaders
  hf_dsets = HF_Datasets(glue_dsets[task], cols=cols, hf_toker=hf_tokenizer, n_inp=n_inp)
  dl_cache_name = cache_name.replace('tokenized', 'dl').replace('.arrow', '.json')
  glue_dls[task] = hf_dsets.dataloaders(bs=32, pad_idx=hf_tokenizer.pad_token_id, cache_name=dl_cache_name)

# %% [markdown]
# # 2. Finetuning model
# %% [markdown]
# * ELECTRA use CLS encodings as pooled result to predict the sentence. (see [here](https://github.com/google-research/electra/blob/79111328070e491b287c307906701ebc61091eb2/model/modeling.py#L254) of its official repository)
# 
# * Note that we should use different prediction head instance for different tasks.

# %%
class SentencePredictHead(nn.Module):
  "The way that Electra and Bert do for sentence prediction task"
  def __init__(self, hidden_size, targ_voc_size):
    super().__init__()
    self.linear = nn.Linear(hidden_size, targ_voc_size)
    self.dropout = nn.Dropout(0.1)
  def forward(self, x):
    "x: (batch size, sequence length, hidden_size)"
    # project the first token (a special token)'s hidden encoding
    return self.linear(self.dropout(x[:,0])).squeeze(-1) # if regression task, squeeze to (B), else (B,#class)

# %% [markdown]
# # 3. Single Task Finetuning
# %% [markdown]
# ## 3.1 Discriminative learning rate

# %%
# Names come from, for nm in model.named_modules(): print(nm[0])

def hf_electra_param_splitter(model, num_hidden_layers, outlayer_name):
  names = ['.embeddings', *[f'encoder.layer.{i}' for i in range(num_hidden_layers)], outlayer_name]
  def end_with_any(name): return any( name.endswith(n) for n in names )
  groups = [ list(mod.parameters()) for name, mod in model.named_modules() if end_with_any(name) ]
  assert len(groups) == len(names)
  return groups

def get_layer_lrs(lr, decay_rate_of_depth, num_hidden_layers):
  # I think input layer as bottom and output layer as top, which make 'depth' mean different from the one of official repo 
  return [ lr * (decay_rate_of_depth ** depth) for depth in reversed(range(num_hidden_layers+2))]

# %% [markdown]
# ## 3.2 Learning rate schedule

# %%
def linear_warmup_and_decay(pct_now, lr_max, end_lr, decay_power, warmup_pct, total_steps):
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

# %% [markdown]
# ## 3.3 finetune

# %%
METRICS = {
  **{ task:['MatthewsCorrCoef'] for task in ['cola']},
  **{ task:['Accuracy'] for task in ['sst2', 'mnli', 'qnli', 'rte', 'wnli', 'snli','ax']},
  # Note: MRPC and QQP are both binary classification problem, so we can just use fastai's default
  # average option 'binary' without spcification of average method.
  **{ task:['F1Score', 'Accuracy'] for task in ['mrpc', 'qqp']}, 
  **{ task:['PearsonCorrCoef', 'SpearmanCorrCoef'] for task in ['stsb']}
}
TARG_VOC_SIZE = {
    **{ task:1 for task in ['stsb']},
    **{ task:2 for task in ['cola', 'sst2', 'mrpc', 'qqp', 'qnli', 'rte', 'wnli']},
    **{ task:3 for task in ['mnli','ax']}
}

# %%
class MyMSELossFlat(BaseLoss):

  def __init__(self,*args, axis=-1, floatify=True, low=None, high=None, **kwargs):
    super().__init__(nn.MSELoss, *args, axis=axis, floatify=floatify, is_2d=False, **kwargs)
    self.low, self.high = low, high

  def decodes(self, x):
    if self.low is not None: x = torch.max(x, x.new_full(x.shape, self.low))
    if self.high is not None: x = torch.min(x, x.new_full(x.shape, self.high))
    return x

# %%
def get_glue_learner(task, one_cycle=False, device='cuda:0', run_name=None, checkpoint=None):
  
  # num_epochs
  if task == 'rte': num_epochs = 10
  else: num_epochs = 3

  # dls
  dls = glue_dls[task].to(torch.device(device))
  # model
  if task=='wnli' and config['use_wsc']:
    model = ELECTRAWSCModel(HF_Model(ElectraForPreTraining, f"google/electra-{SIZE}-discriminator", hf_tokenizer))
  else:
    model = nn.Sequential(HF_Model(ElectraModel, f"google/electra-{SIZE}-discriminator", hf_tokenizer),
                          SentencePredictHead(electra_config.hidden_size, targ_voc_size=TARG_VOC_SIZE[task]))
  # loss func
  if task == 'stsb': loss_fc = MSELossFlat()
  elif task=='wnli' and config['use_wsc']: loss_fc = BCEWithLogitsLossFlat()
  else: loss_fc = CrossEntropyLossFlat()
  # metrics
  metrics = [eval(f'{metric}()') for metric in METRICS[task]]
  def sigmoid_acc(inps,targ):
    pred = torch.sigmoid(inps) > 0.5
    return (pred == targ).float().mean()
  if task=='wnli' and config['use_wsc']: metrics = [sigmoid_acc]
  # learning rate
  splitter = partial(hf_electra_param_splitter, 
                  num_hidden_layers=electra_config.num_hidden_layers,
                  outlayer_name= 'discriminator_predictions' if task=='wnli' and config['use_wsc'] else '1')
  layer_lrs = get_layer_lrs(lr=config['lr'],
                    decay_rate_of_depth=config['layer_lr_decay'],
                    num_hidden_layers=electra_config.num_hidden_layers,)
  lr_shedule = ParamScheduler({'lr': partial(linear_warmup_and_decay,
                                            lr_max=np.array(layer_lrs),
                                            end_lr=0.0,
                                            decay_power=1,
                                            warmup_pct=0.1,
                                            total_steps=num_epochs*(len(dls.train)))})
  
  
  # learner
  learn = Learner(dls, model,
                  loss_func=loss_fc, 
                  opt_func=partial(Adam, eps=1e-6,),
                  metrics=metrics,
                  splitter=splitter,
                  lr=layer_lrs,
                  path=str(Path.home()/'checkpoints'),
                  model_dir='electra_glue',)
  
  # load checkpoint
  if checkpoint: learn.load(checkpoint)

  # fp16
  if config['use_fp16']: learn = learn.to_fp16()

  # 
  if run_name:
    id = run_name.split('_')[1]
    wandb.init(project='electra-glue', name=run_name, config={'task': task, 'id':id, 'use_fp16':config['use_fp16'], 'optim':'Adam', 'use_onecycle':False}, reinit=True)
    learn.add_cb(WandbCallback(None, False))

  # one cycle / warm up + linear decay 
  if one_cycle: return learn, partial(learn.fit_one_cycle, n_epoch=num_epochs)
  else: return learn, partial(learn.fit, n_epoch=num_epochs, cbs=[lr_shedule])


# %%
rand_id = random.randint(1,500)
#rand_id = 79
pretrained_checkpoint = Path.home()/'checkpoints/electra_pretrain/7-06_10-31-49_100%.pth'
pretrained_checkpoint = None
for i in range(10):
  for task in ['cola', 'sst2', 'mrpc', 'stsb', 'qnli', 'rte', 'qqp', 'mnli', 'wnli']:
    if task not in ['wnli']: continue
    run_name = f"{task}_{rand_id}_{i}"
    # run_name = None # set to None to skip wandb and model saving
    learn, fit_fc = get_glue_learner(task, device='cuda:0', 
                                      run_name=run_name, checkpoint=pretrained_checkpoint) 
    fit_fc()
    if run_name:
      wandb.join()
      learn.save(run_name)