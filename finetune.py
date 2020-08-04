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
from transformers.modeling_electra import ElectraClassificationHead
from fastai2.text.all import *
import wandb
from fastai2.callback.wandb import WandbCallback
from _utils.would_like_to_pr import *
from _utils.huggingface import *
from _utils.utils import *


# %%
c = MyConfig({
  # device to train and test
  'device': 'cuda:3', # specify List[int] to use multi gpu (data parallel), None for using all gpu devices
  # the id that in the name of runs which are in the same group (to choose the best from 10 runs and ..)
  'group_id': 172119125, #random.randint(1,999), # None to not save checkpoint and not create wandb run 
  # checkpoint of ELECTRA from pretrain.py, note that only discriminator part of it will be extracted and used
  'pretrained_checkpoint': Path.home()/'checkpoints/electra_pretrain/8-03_17-21-19_12.5%.pth', # None to use checkpoints hubbed on Huggingface
  'head': 'slp',
  # whether to use wnli trick described in the paper
  'wsc_trick': True,
  # size of ELECTRA
  'size': 'small',
  'max_length': 128, # 128 only if ELECTRA-small (note that all public models are ++model which use 512)
  # cache the data under it
  'cache_dir': Path.home()/'datasets', # ! should be `pathlib.Path` istance
  # where to save finetuneing checkpoints
  'ckp_dir': Path.home()/'checkpoints/electra_glue', # ! should be `pathlib.Path` istance
  # whether to do finetune or test
  'do_finetune': True, # True, to do only finetuning, False to do only test
  # cache_dir/glue/test/out_dir to put output files of testing.
  'out_dir': "hf_small_test" ,
  # finetuning checkpoint for testing. These will become "ckp_dir/{task}_{group_id}_{th_run}.pth"
  'th_run': {'cola': 9, 'sst2': 1, 'mrpc': 6, 'qqp': 2, 'stsb': 4, 'qnli': 1, 'rte': 4, 'mnli': 5, 'ax': 5,
             'wnli': [22,3,10,14,8,0,17,20,2,6],
            }
})
assert c.head in ['slp', 'mlp', 'mlp_load']
if c.size == 'small' and c.pretrained_checkpoint: assert c.max_length == 128, "Make sure max_length is 128 for ELECTRA-small, or comment this line if you know what you are doing."
if c.pretrained_checkpoint is None: assert c.max_length == 512, "All public models of ELECTRA is ++, and use max_length 512 when finetuning on GLUE"
if c.wsc_trick:
  from _utils.wsc_trick import * # importing spacy model takes time
if c.size == 'small': c.lr = 3e-4; c.layer_lr_decay = 0.8
elif c.size == 'base': c.lr = 1e-4; c.layer_lr_decay = 0.8
elif c.size == 'large': c.lr = 5e-5; c.layer_lr_decay = 0.9
else: raise ValueError(f"Invalid size {c.size}")
hf_tokenizer = ElectraTokenizerFast.from_pretrained(f"google/electra-{c.size}-discriminator")
electra_config = ElectraConfig.from_pretrained(f'google/electra-{c.size}-discriminator')
c.cache_dir.mkdir(parents=True, exist_ok=True)

# %% [markdown]
# # 1. Prepare data
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
  if task == 'wnli' and c.wsc_trick:
    benchmark, subtask = 'super_glue', 'wsc'
    # samples in all splits are all less than 128-2, so don't need to worry about max_length
    Tfm = partial(WSCTrickTfm, hf_toker=hf_tokenizer)
    cols = {'prefix':TensorText, 'suffix':TensorText, 'cands':TensorText, 'cand_lens':noop, 'label':TensorCategory}
    n_inp=4
    cache_name = "tricked_{split}.arrow"
  else:
    benchmark, subtask = 'glue', task
    tok_func = partial(tokenize_sents_max_len, cols=textcols(task), max_length=c.max_length)
    Tfm = partial(HF_Transform, func=tok_func)
    cols = ['inp_ids', 'label']
    n_inp=1
    cache_name = f"tokenized_{c.max_length}_{{split}}.arrow"
  # load / download datasets.
  dsets = nlp.load_dataset(benchmark, subtask, cache_dir=c.cache_dir)
  # There is two samples broken in QQP training set
  if task=='qqp': dsets['train'] = dsets['train'].filter(lambda e: e['question2']!='',
                                          cache_file_name=str(c.cache_dir/'glue/qqp/1.0.0/fixed_train.arrow'))
  # 
  if task=='wnli' and c.wsc_trick: dsets['train'] = dsets['train'].filter(lambda e: e['label']==1,
                 cache_file_name=str(c.cache_dir/'super_glue/wsc/1.0.2/filtered_tricked_train.arrow'))
  # load / make tokenized datasets
  glue_dsets[task] = Tfm(dsets).map(cache_name=cache_name)
  # load / make dataloaders
  hf_dsets = HF_Datasets(glue_dsets[task], cols=cols, hf_toker=hf_tokenizer, n_inp=n_inp)
  dl_cache_name = cache_name.replace('tokenized', 'dl').replace('.arrow', '.json')
  glue_dls[task] = hf_dsets.dataloaders(bs=32, cache_name=dl_cache_name)

# %% [markdown]
# ## 1.2 View Data
# - View raw data on [nlp-viewer]! (https://huggingface.co/nlp/viewer/)
# 
# - View task description on Tensorflow dataset doc for GLUE (https://www.tensorflow.org/datasets/catalog/glue) 
# 
# - You may notice some text without \[SEP\], that is because the whole sentence is truncated by `show_batch`, you can turn it off by specify `truncated_at=None`
# %% [markdown]
# 

# %%
# CoLA (The Corpus of Linguistic Acceptability) - 0: unacceptable, 1: acceptable 
print("Dataset size (train/valid/test): {}/{}/{}".format(*[len(dl.dataset) for dl in glue_dls['cola'].loaders]))
glue_dls['cola'].show_batch(max_n=1)

# %% [markdown]
# 
# %% [markdown]
# 

# %%
# SST-2 (The Stanford Sentiment Treebank) - 1: positvie, 0: negative
print("Dataset size (train/valid/test): {}/{}/{}".format(*[len(dl.dataset) for dl in glue_dls['sst2'].loaders]))
glue_dls['sst2'].show_batch(max_n=1)


# %%
# MRPC (Microsoft Research Paraphrase Corpus) -  1: match, 0: no
print("Dataset size (train/valid/test): {}/{}/{}".format(*[len(dl.dataset) for dl in glue_dls['mrpc'].loaders]))
glue_dls['mrpc'].show_batch(max_n=1)

# %% [markdown]
# 

# %%
# STS-B (Semantic Textual Similarity Benchmark) - 0.0 ~ 5.0
print("Dataset size (train/valid/test): {}/{}/{}".format(*[len(dl.dataset) for dl in glue_dls['stsb'].loaders]))
glue_dls['stsb'].show_batch(max_n=1)


# %%
# QQP (Quora Question Pairs) - 0: no, 1: duplicated
print("Dataset size (train/valid/test): {}/{}/{}".format(*[len(dl.dataset) for dl in glue_dls['qqp'].loaders]))
glue_dls['qqp'].show_batch(max_n=1)


# %%
# MNLI (The Multi-Genre NLI Corpus) - 0: entailment, 1: neutral, 2: contradiction
print("Dataset size (train/validation_matched/validation_mismatched/test_matched/test_mismatched): {}/{}/{}/{}/{}".format(*[len(dl.dataset) for dl in glue_dls['mnli'].loaders]))
glue_dls['mnli'].show_batch(max_n=1)


# %%
# QNLI (The Stanford Question Answering Dataset) - 0: entailment, 1: not_entailment
print("Dataset size (train/valid/test): {}/{}/{}".format(*[len(dl.dataset) for dl in glue_dls['qnli'].loaders]))
glue_dls['qnli'].show_batch(max_n=1)


# %%
# RTE (Recognizing_Textual_Entailment) - 0: entailment, 1: not_entailment
print("Dataset size (train/valid/test): {}/{}/{}".format(*[len(dl.dataset) for dl in glue_dls['rte'].loaders]))
glue_dls['rte'].show_batch(max_n=1)


# %%
# WSC (The Winograd Schema Challenge) - 0: wrong, 1: correct
# There are three style, WNLI (casted in NLI type), WSC, WSC with candidates (trick used by Roberta)
"Note for WSC trick: cands is the concatenation of candidates, cand_lens is the lengths of candidates in order."
print("Dataset size (train/valid/test): {}/{}/{}".format(*[len(dl.dataset) for dl in glue_dls['wnli'].loaders]))
glue_dls['wnli'].show_batch(max_n=1)


# %%
# AX (GLUE Diagnostic Dataset) - 0: entailment, 1: neutral, 2: contradiction
print("Dataset size (test): {}".format(*[len(dl.dataset) for dl in glue_dls['ax'].loaders]))
glue_dls['ax'].show_batch(max_n=1)

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
def get_glue_learner(task, run_name=None, one_cycle=False, inference=False):
  
  # num_epochs
  if task == 'rte': num_epochs = 10
  else: num_epochs = 3

  # dls
  dls = glue_dls[task]
  if isinstance(c.device, str): dls.to(torch.device(c.device))
  elif isinstance(c.device, list): dls.to(torch.device('cuda', c.device[0]))
  else: dls.to(torch.device('cuda:0'))
  # load model
  if c.pretrained_checkpoint is not None: 
    discriminator = HF_Model(ElectraForPreTraining, electra_config, hf_tokenizer)
    if c.pretrained_checkpoint:
      load_part_model(c.pretrained_checkpoint, discriminator, 'discriminator')
  else:
    discriminator = HF_Model(ElectraForPreTraining, f"google/electra-{c.size}-discriminator", hf_tokenizer)
  # model
  if task=='wnli' and c.wsc_trick:
    model = ELECTRAWSCTrickModel(discriminator, hf_tokenizer.pad_token_id)
  elif c.head=='slp': # take only base model and mount an output layer
    model = nn.Sequential(HF_Model(discriminator.model.electra, hf_toker=hf_tokenizer), 
                          SentencePredictHead(electra_config.hidden_size, targ_voc_size=TARG_VOC_SIZE[task]))
  else:
    _config = deepcopy(electra_config)
    _config.num_labels = TARG_VOC_SIZE[task]
    if c.head=='mlp_load':
      cls_head = ElectraClassificationHead(_config)
      cls_head.dense = discriminator.model.discriminator_predictions.dense
    model = nn.Sequential(HF_Model(discriminator.model.electra, hf_toker=hf_tokenizer), 
                          cls_head)
  # loss func
  if task == 'stsb': loss_fc = MyMSELossFlat(low=0.0, high=5.0)
  elif task=='wnli' and c.wsc_trick: loss_fc = ELECTRAWSCTrickLoss()
  else: loss_fc = CrossEntropyLossFlat()
  # metrics
  metrics = [eval(f'{metric}()') for metric in METRICS[task]]
  if task=='wnli' and c.wsc_trick: metrics = [accuracy_electra_wsc_trick]
  # learning rate
  splitter = partial(hf_electra_param_splitter, 
                     num_hidden_layers=electra_config.num_hidden_layers,
                     outlayer_name= 'discriminator_predictions' if task=='wnli' and c.wsc_trick else '1')
  layer_lrs = get_layer_lrs(lr=c.lr,
                    decay_rate_of_depth=c.layer_lr_decay,
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
                  opt_func=partial(Adam, eps=1e-6, wd=0.),
                  metrics=metrics,
                  splitter=splitter if not inference else trainable_params,
                  lr=layer_lrs if not inference else defaults.lr,
                  path=str(c.ckp_dir.parent),
                  model_dir=c.ckp_dir.name,)
  
  # multi gpu
  if isinstance(c.device, list) or c.device is None:
    learn.model = nn.DataParallel(learn.model, device_ids=c.device)

  # fp16
  #if c.device != 'cpu': learn = learn.to_fp16(clip=1.)

  # wandb
  if run_name:
    wandb.init(project='electra-glue', name=run_name, config={'task': task, 'optim':'Adam', **c}, reinit=True)
    learn.add_cb(WandbCallback(None, False))

  # one cycle / warm up + linear decay 
  if one_cycle: return learn, partial(learn.fit_one_cycle, n_epoch=num_epochs)
  else: return learn, partial(learn.fit, n_epoch=num_epochs, cbs=[lr_shedule])


# %%
if c.do_finetune:
  for i in range(5,10):
    for task in ['cola', 'sst2', 'mrpc', 'stsb', 'qnli', 'rte', 'qqp', 'mnli', 'wnli']:
      if task in ['wnli']: continue # to only do some tasks
      if c.group_id: run_name = f"{task}_{c.group_id}_{i}";
      else: run_name = None; print(task)
      learn, fit_fc = get_glue_learner(task, run_name)
      #bk()
      fit_fc()
      if run_name:
        wandb.join()
        learn.save(run_name)

# %% [markdown]
# ## 3.3 Predict the test set

# %%
def get_identifier(task, split):
  map = {'cola': 'CoLA', 'sst2':'SST-2', 'mrpc':'MRPC', 'qqp':'QQP', 'stsb':'STS-B', 'qnli':'QNLI', 'rte':'RTE', 'wnli':'WNLI', 'ax':'AX'}
  if task =='mnli' and split == 'test_matched': return 'MNLI-m'
  elif task == 'mnli' and split == 'test_mismatched': return 'MNLI-mm'
  else: return map[task]

class Ensemble(nn.Module):
  def __init__(self, models, device='cuda:0', merge_out_fc=None):
    super().__init__()
    self.models = nn.ModuleList( m.cpu() for m in models )
    self.device = device
    self.merge_out_fc = merge_out_fc
  
  def to(self, device): 
    self.device = device
    return self
  def getitem(self, i): return self.models[i]
  
  def forward(self, *args, **kwargs):
    outs = []
    for m in self.models:
      m.to(self.device)
      out = m(*args, **kwargs)
      m.cpu()
      outs.append(out)
    if self.merge_out_fc:
      outs = self.merge_out_fc(outs)
    else:
      outs = torch.stack(outs)
      outs = outs.mean(dim=0)
    return outs

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


# %%
def predict_test(task, checkpoint, dl_idx=-1, output_dir=None, device='cuda:0'):
  if output_dir is None: output_dir = c.cache_dir/'glue/test'
  output_dir = Path(output_dir)
  output_dir.mkdir(exist_ok=True)
  device = torch.device(device)

  # load checkpoint and get predictions
  learn, _ = get_glue_learner(task, device=device, inference=True)
  if task == 'wnli' and config['wsc_trick']:
    load_model_(learn, checkpoint, merge_out_fc=wsc_trick_merge)
  else:
    load_model_(learn, checkpoint)
  results = learn.get_preds(ds_idx=dl_idx, with_decoded=True)
  preds = results[-1] # preds -> (predictions logits, targets, decoded prediction)

  # decode target class index to its class name 
  if task in ['mnli','ax']:
    preds = [ ['entailment','neutral','contradiction'][p] for p in preds]
  elif task in ['qnli','rte']: 
    preds = [ ['entailment','not_entailment'][p] for p in preds ]
  elif task == 'wnli' and c.wsc_trick:
    preds = preds.to(dtype=torch.long).tolist()
  else: preds = preds.tolist()
    
  # form test dataframe and save
  test_df = pd.DataFrame( {'index':range(len(list(glue_dsets[task].values())[dl_idx])), 'prediction': preds} )
  split = list(glue_dsets['mnli'].keys())[dl_idx]
  identifier = get_identifier(task, split)
  test_df.to_csv( output_dir/f'{identifier}.tsv', sep='\t' )
  return test_df


# %%
if not c.do_finetune:
  for task, th in c.th_run.items():
    if task not in ['wnli']: continue # to do only some task
    print(task)
    # ax use mnli ckp
    if isinstance(th, int):
      ckp = f"{task}_{c.group_id}_{th}" if task != 'ax' else f"mnli_{c.group_id}_{th}"
    else:
      ckp = [f"{task}_{c.group_id}_{i}" if task != 'ax' else f"mnli_{c.group_id}_{i}" for i in th]
    # run test for all testset in this task
    dl_idxs = [-1, -2] if task=='mnli' else [-1]
    for dl_idx in dl_idxs:
      df = predict_test(task, ckp, dl_idx, output_dir=c.cache_dir/f'glue/test/{c.out_dir}')


