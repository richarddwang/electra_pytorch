import torch
from torch import nn
from fastai2.text.all import *
from _utils.huggingface import HF_BaseTransform, HF_Model

class WSCTransform(HF_BaseTransform):

  def __init__(self, hf_dset, hf_toker, **kwargs):
    super().__init__(hf_dset, out_cols=['inp_ids', 'span', 'label'], **kwargs)
    self.tokenizer = hf_toker
    self.tokenizer_config = hf_toker.pretrained_init_configuration

  def __call__(self, sample):
    # get texts without target span
    target_start, target_end = sample['span2_index'], sample['span2_index']+len(sample['span2_text'].split())
    prefix = self.tokenizer.encode(' '.join(sample['text'].split()[:target_start]))[:-1] # no sep
    suffix = self.tokenizer.encode(' '.join(sample['text'].split()[target_end:]))[1:] # no cls
    candidate = self.tokenizer.encode(sample['span1_text'])[1:-1] # no cls and sep
    sample['inp_ids'] = prefix + candidate + suffix
    sample['span'] = [len(prefix), len(prefix)+len(candidate)] # start and end of candidate
    return sample

  def __getstate__(self):
    state = self.__dict__.copy() 
    state['tokenizer'] = None 
    return state

class ELECTRAWSCModel(nn.Module):
  def __init__(self, discriminator):
    super().__init__()
    self.model = discriminator
  @classmethod
  def from_hf(cls, *args,**kwargs):
    self.model = HF_Model(*args,**kwargs)
  def forward(self, x, spans):
    scores = self.model(x) # (B, L)
    masks = torch.stack([x.new_zeros(x.shape[1]).scatter(0,torch.arange(start,end,device=x.device),1) for start,end in spans])
    return (scores * masks).sum(dim=-1) / masks.sum(dim=-1) # (B,)