from functools import reduce
import operator
import en_core_web_lg # python -m spacy download en_core_web_lg
spnlp = en_core_web_lg.load()
import torch
from fastai2.text.all import *
from _utils.huggingface import HF_BaseTransform

@delegates(but=["out_cols"])
class WSCTrickTfm(HF_BaseTransform):

  def __init__(self, hf_dset, hf_toker, extract_candidates=True, exclude_false_sample=False, **kwargs):
    super().__init__(hf_dset, out_cols=['inp_ids', 'span', 'label'], **kwargs)
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
    sample['cand_lens'] = [len(cand) for cand in cands]
    sample['cands'] = reduce(operator.add, cands, []) # flatten list, into a 1d token ids

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

def pad_all_input_chunk(samples, n, pad_idx, seq_len=72):
  "Pad n element to the first of `samples` by adding padding by chunks of size `seq_len`"
  max_lens = [max([len(s[i]) for s in samples]) for i in range(n)]
  def _f(x, max_len):
    l = max_len - x.shape[0]
    pad_chunk = x.new_zeros((l//seq_len) * seq_len) + pad_idx
    pad_res   = x.new_zeros(l % seq_len) + pad_idx
    x1 = torch.cat([x, pad_res, pad_chunk])
    return retain_type(x1, x)
  return [(*( _f(s[i], max_len) for i, max_len in enumerate(max_lens)), *s[n:]) for s in samples]

class WSCTrickCB(Callback):
  def __init__(self, pad_idx):
    self.pad_idx = pad_idx
  
  def begin_batch(self):
    batch_size = len(self.xb[0])
    scores = []; max_n_cand = 0
    for i in range(batch_size):
      prefix, suffix, cands, cand_lens = self.depad(self.xb[0][i]), self.depad(self.xb[1][i]), self.depad(self.xb[2][i]), self.depad(self.xb[3][i])
      # split into list of tensors, ith has length of cand_lens[i]
      cands = cands.split(cand_lens.tolist())
      if len(cands) > max_n_cand: max_n_cand = len(cands)
      max_len = max(len(cand) for cand in cands) + len(prefix) + len(suffix)
      sents, masks = [], []
      for cand in cands:
        pad_len = max_len - len(prefix) - len(suffix) - len(cand)
        sents.append( torch.cat([prefix,cand,suffix,cand.new_full((pad_len,),self.pad_idx)]) )
        masks.append( torch.cat([cand.new_zeros(len(prefix)),cand.new_ones(len(cand)),cand.new_zeros(max_len-len(prefix)-len(cand))]) )
      sents = torch.stack(sents)
      masks = torch.stack(masks)
      logits = self.model(sents) # (#candidate, L, vocab_size)
      lprobs = F.log_softmax(logits, dim=-1, dtype=torch.float) # (#candidate, L, vocab_size)
      # choose only the probs related to tokens
      _scores = lprobs.gather(dim=-1, index=sents.unsqueeze(-1)).squeeze(-1) # (#candidate, L)
      # average scores for the candidate span
      #bk()
      _scores = (_scores * masks).sum(dim=-1) / masks.sum(dim=-1) # (#candidate,)
      scores.append(_scores)
    scores = [torch.cat([_scores, _scores.new_full((max_n_cand-len(_scores),), -float('inf'))]) for _scores in scores]
    scores = torch.stack(scores) # (B, max_n_cand)

    self.learn.pred = scores;                                                self.learn('after_pred')
    if len(self.learn.yb) == 0: raise CancelBatchException
    self.learn.loss = self.learn.loss_func(self.learn.pred, *self.learn.yb); self.learn('after_loss')
    if not self.learn.training: raise CancelBatchException
    self.learn.loss.backward();                                              self.learn('after_backward')
    self.learn.opt.step();                                                   self.learn('after_step')
    self.learn.opt.zero_grad()
    raise CancelBatchException

  def depad(self, tensor):
    mask = tensor != self.pad_idx
    return tensor.masked_select(mask)

class WSCTrickLoss():
  def __call__(self, inp, targ):
    return F.cross_entropy(inp, targ)
  
  def decodes(self, scores):
    # scores: list of (n_cand)
    scores = self._batchize(scores) # (n_sample, max_num_cand)
    query_scores = scores[:,0].unsqueeze(-1) # (n_sample,1)
    labels = (scores[:,1:] < query_scores).all(dim=1) # (n_sample,)
    return labels
  
  def _batchize(self, tensors):
    max_len = max([len(t) for t in tensors])
    tensors = [torch.cat([t,t.new_full((max_len-len(t),), -float('inf'))]) for t in tensors]
    return torch.stack(tensors)