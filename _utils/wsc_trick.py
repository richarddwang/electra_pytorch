from functools import reduce
import operator
import en_core_web_lg # python -m spacy download en_core_web_lg
spnlp = en_core_web_lg.load()
import torch
from fastai2.text.all import *
from _utils.huggingface import HF_BaseTransform, HF_Model

@delegates(but=["out_cols"])
class WSCTrickTfm(HF_BaseTransform):

  def __init__(self, hf_dset, hf_toker, **kwargs):
    super().__init__(hf_dset, out_cols=['prefix', 'suffix', 'cands', 'cand_lens', 'label'], **kwargs)
    self.tokenizer = hf_toker
    self.tokenizer_config = hf_toker.pretrained_init_configuration

  def __getstate__(self):
    state = self.__dict__.copy()
    state['tokenizer'] = None
    return state

  def simple_encode(self, text):
    return self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(text))

  def __call__(self, sample):
    # get candidates that solve pronoun
    sentence = spnlp(sample['text'])
    noun_chunks = extended_noun_chunks(sentence)
    cand_spans = filter_noun_chunks(
                    noun_chunks,
                    exclude_pronouns=True,
                    exclude_query=sample['span1_text'],
                    exact_match=False,)
    cands = [str(span) for span in cand_spans]
    cands = list(set(cands)) # no repeated

    # get texts without target span
    target_start, target_end = sample['span2_index'], sample['span2_index']+len(sample['span2_text'].split())
    prefix = ' '.join(sample['text'].split()[:target_start])
    suffix = ' '.join(sample['text'].split()[target_end:])

    # tokenize
    sample['prefix'] = self.tokenizer.encode(prefix)[:-1] # no SEP
    sample['suffix'] = self.tokenizer.encode(suffix)[1:] # no CLS
    cands = [self.simple_encode(sample['span1_text'])] + [self.simple_encode(cand) for cand in cands]
    sample['cands'] = reduce(operator.add, cands, []) # flatten list, into a 1d token ids
    sample['cand_lens'] = [len(cand) for cand in cands]
    # sample already have 'label'

    return sample

def extended_noun_chunks(sentence):
    noun_chunks = {(np.start, np.end) for np in sentence.noun_chunks}
    np_start, cur_np = 0, 'NONE'
    for i, token in enumerate(sentence):
        np_type = token.pos_ if token.pos_ in {'NOUN', 'PROPN'} else 'NONE'
        if np_type != cur_np:
            if cur_np != 'NONE':
                noun_chunks.add((np_start, i))
            if np_type != 'NONE':
                np_start = i
            cur_np = np_type
    if cur_np != 'NONE':
        noun_chunks.add((np_start, len(sentence)))
    return [sentence[s:e] for (s, e) in sorted(noun_chunks)]

def filter_noun_chunks(chunks, exclude_pronouns=False, exclude_query=None, exact_match=False):
    if exclude_pronouns:
        chunks = [
            np for np in chunks if (
                np.lemma_ != '-PRON-'
                and not all(tok.pos_ == 'PRON' for tok in np)
            )
        ]

    if exclude_query is not None:
        excl_txt = [exclude_query.lower()]
        filtered_chunks = []
        for chunk in chunks:
            lower_chunk = chunk.text.lower()
            found = False
            for excl in excl_txt:
                if (
                    (not exact_match and (lower_chunk in excl or excl in lower_chunk))
                    or lower_chunk == excl
                ):
                    found = True
                    break
            if not found:
                filtered_chunks.append(chunk)
        chunks = filtered_chunks

    return chunks

class ELECTRAWSCTrickModel(nn.Module):
  def __init__(self, discriminator, pad_idx):
    super().__init__()
    self.model = discriminator
    self.pad_idx = pad_idx
  
  def forward(self, *xb):
    """
    prefix: (B, L_p)
    suffix: (B, L_s)
    cands: (B, L_c)
    cand_lens: (B, L_cl)
    """
    batch_size = xb[0].shape[0]

    all_scores = []
    n_cands = []
    for i in range(batch_size):
      # unpad
      prefix, suffix, cands, cand_lens = self.depad(xb[0][i]), self.depad(xb[1][i]), self.depad(xb[2][i]), self.depad(xb[3][i])
      # unpack and pad into (#candidate, max_len)
      cands = cands.split(cand_lens.tolist()) # split into list of tensors, ith has length of cand_lens[i]
      max_len = max(len(cand) for cand in cands) + len(prefix) + len(suffix)
      sents, masks = [], []
      for cand in cands:
        pad_len = max_len - len(prefix) - len(suffix) - len(cand)
        sents.append( torch.cat([prefix,cand,suffix,cand.new_full((pad_len,),self.pad_idx)]) )
        masks.append( torch.cat([cand.new_zeros(len(prefix)),cand.new_ones(len(cand)),cand.new_zeros(max_len-len(prefix)-len(cand))]) )
      sents = torch.stack(sents) # (#candidate, max_len)
      masks = torch.stack(masks) # (#candidate, max_len)
      # get discrimiator scores for each candidate
      logits = self.model(sents) # (#candidate, max_len)
      scores = (logits * masks).sum(dim=-1) # (#candidate,)
      scores = scores / masks.sum(dim=-1)
      # save
      all_scores.append(scores)
      n_cands.append(scores.shape[0])
    # repack
    all_scores = torch.cat(all_scores) # (#total candidate in this batch,)
    n_cands = torch.tensor(n_cands, device=all_scores.device) # (B,)
    return all_scores, n_cands

  def depad(self, tensor):
    mask = tensor != self.pad_idx
    return tensor.masked_select(mask)

def wsc_trick_predict(preds):
  """
  all_scores: (#total candidates in the dataset,)
  n_cands: (#samples in the dataset,)
  """
  all_scores, n_cands = preds
  predicted = []
  for scores in all_scores.split(n_cands.tolist()):
    query_score  = scores[0]
    other_scores = scores[1:]
    predicted.append((query_score <= other_scores).all())
  return torch.stack(predicted).int()

class ELECTRAWSCTrickLoss():
  def __init__(self):
    self.criterion = nn.BCEWithLogitsLoss()
  
  def __call__(self, x, y):
    all_scores, n_cands = x
    all_labels = []
    for scores in all_scores.split(n_cands.tolist()):
      n_cand = len(scores)
      labels = scores.new_ones(n_cand)
      labels[0] = 0.
      all_labels.append(labels)
    all_labels = torch.cat(all_labels) # (#total candidate in this batch,)
    return self.criterion(all_scores, all_labels)

  def decodes(self, preds): return wsc_trick_predict(preds)

def accuracy_electra_wsc_trick(preds, targs):
  predicts = wsc_trick_predict(preds)
  return (predicts == targs).float().mean()

def wsc_trick_merge(outs):
  all_scores = torch.stack([out[0] for out in outs]).mean(dim=0)
  n_cands = outs[0][1]
  return all_scores, n_cands