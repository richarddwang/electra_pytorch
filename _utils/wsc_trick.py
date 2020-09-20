from functools import reduce
import operator
import en_core_web_lg # python -m spacy download en_core_web_lg
spnlp = en_core_web_lg.load()
import torch
from fastai.text.all import *

def wsc_trick_process(sample, hf_toker):
  
  # clean up the query
  query = sample['span1_text']
  query = query.replace('\n',' ')
  if query.endswith('.') or query.endswith(','):
    query = query[:-1]

  # split tokens
  tokens = sample['text'].split(' ')

  # find the pronoun
  def strip_pronoun(x): return x.rstrip('.,"')
  pronoun_idx = sample['span2_index']
  pronoun = strip_pronoun(sample['span2_text'])
  if strip_pronoun(tokens[pronoun_idx]) != pronoun:
      # hack: sometimes the index is misaligned
      if strip_pronoun(tokens[pronoun_idx + 1]) == pronoun:
          pronoun_idx += 1
      else:
          raise Exception('Misaligned pronoun!')
  assert strip_pronoun(tokens[pronoun_idx]) == pronoun

  # split tokens before and after the pronoun
  before = tokens[:pronoun_idx]
  after = tokens[pronoun_idx + 1:]

  # detokenize
  before = ' '.join(before)
  after = ' '.join(after)

  # space
  leading_space = ' ' if pronoun_idx > 0 else ''
  trailing_space = ' ' if len(after) > 0 else ''

  # hack: when the pronoun ends in a period (or comma), move the
  # punctuation to the "after" part
  if pronoun.endswith('.') or pronoun.endswith(','):
      after = pronoun[-1] + ' ' + after
      pronoun = pronoun[:-1]
  
  # hack: when the "after" part begins with a comma or period, remove
  # the trailing space
  if after.startswith('.') or after.startswith(','):
      trailing_space = ''

  # parse sentence with spacy
  sentence = spnlp(before + leading_space + pronoun + trailing_space + after)

  # find pronoun span
  start = len(before + leading_space)
  first_pronoun_tok = find_token(sentence, start_pos=start)
  pronoun_span = find_span(sentence, pronoun, start=first_pronoun_tok.i)
  assert pronoun_span.text == pronoun

  label = sample['label']

  prefix = sentence[:pronoun_span.start].text
  suffix = sentence[pronoun_span.end:].text_with_ws

   # spaCy spans include trailing spaces, but we need to know about
  leading_space = ' ' if sentence[:pronoun_span.start].text_with_ws.endswith(' ') else ''
  trailing_space = ' ' if pronoun_span.text_with_ws.endswith(' ') else ''

  cand_spans = filter_noun_chunks(
                extended_noun_chunks(sentence),
                exclude_pronouns=True,
                exclude_query=query,
                exact_match=False,
            )

  cands = [str(span) for span in cand_spans]
  cands = list(set(cands)) # no repeated
  
  # tokenize
  def simple_encode(text):
    return hf_toker.convert_tokens_to_ids(hf_toker.tokenize(text))
  sample['prefix'] = hf_toker.encode(prefix)[:-1] # no SEP
  sample['suffix'] = hf_toker.encode(suffix)[1:] # no CLS
  cands = [simple_encode(query)] + [simple_encode(cand) for cand in cands]
  sample['cands'] = reduce(operator.add, cands, []) # flatten list, into a 1d token ids
  sample['cand_lens'] = [len(cand) for cand in cands]
  # sample already have 'label'

  return sample

class ELECTRAWSCTrickModel(nn.Module):
  def __init__(self, discriminator, pad_idx):
    super().__init__()
    self.discriminator = discriminator
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
      cands = cands.split(cand_lens.tolist()) # split into list of tensors, i th has length of cand_lens[i]
      max_len = max(len(cand) for cand in cands) + len(prefix) + len(suffix)
      sents, masks = [], []
      for cand in cands:
        pad_len = max_len - len(prefix) - len(suffix) - len(cand)
        sents.append( torch.cat([prefix,cand,suffix,cand.new_full((pad_len,),self.pad_idx)]) )
        masks.append( torch.cat([cand.new_zeros(len(prefix)),cand.new_ones(len(cand)),cand.new_zeros(max_len-len(prefix)-len(cand))]) )
      sents = torch.stack(sents) # (#candidate, max_len)
      masks = torch.stack(masks) # (#candidate, max_len)
      # get discrimiator scores for each candidate
      logits = self.discriminator(sents)[0] # (#candidate, max_len)
      scores = (logits * masks).sum(dim=-1) # (#candidate,)
      scores = scores / masks.sum(dim=-1)
      # save
      all_scores.append(scores)
      n_cands.append(scores.shape[0])
    # repack
    all_scores = torch.cat(all_scores) # (total number of candidates in this batch,)
    n_cands = torch.tensor(n_cands, device=all_scores.device) # (B,), number of candidates for each sample
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
  for scores in all_scores.split(n_cands.tolist()): # for every sample
    query_score  = scores[0]
    other_scores = scores[1:]
    predicted.append((query_score <= other_scores).all())
  return all_scores.new(predicted)

class ELECTRAWSCTrickLoss():
  def __init__(self):
    self.criterion = nn.BCEWithLogitsLoss()
  
  def __call__(self, x, y):
    all_scores, n_cands = x
    losses = []
    for scores, label in zip(all_scores.split(n_cands.tolist()), y): # for every sample
      if label == 0: continue # only calculate loss on positive samples
      labels = [0.] + [1.]*(len(scores)-1) # labels for BCEWithLogitLoss
      loss = self.criterion(scores, scores.new(labels)) # ?
      losses.append(loss)
    if losses:
      return torch.stack(losses).sum() / len(losses) # average
    else:
      raise CancelBatchException

  def decodes(self, preds): return wsc_trick_predict(preds)

def wsc_trick_accuracy(preds, targs):
  predicts = wsc_trick_predict(preds)
  return (predicts == targs).float().mean()

def wsc_trick_merge(outs):
  all_scores = torch.stack([out[0] for out in outs]).mean(dim=0)
  n_cands = outs[0][1]
  return all_scores, n_cands

def find_token(sentence, start_pos):
    found_tok = None
    for tok in sentence:
        if tok.idx == start_pos:
            found_tok = tok
            break
    return found_tok


def find_span(sentence, search_text, start=0):
    search_text = search_text.lower()
    for tok in sentence[start:]:
        remainder = sentence[tok.i:].text.lower()
        if remainder.startswith(search_text):
            len_to_consume = len(search_text)
            start_idx = tok.idx
            for next_tok in sentence[tok.i:]:
                end_idx = next_tok.idx + len(next_tok.text)
                if end_idx - start_idx == len_to_consume:
                    span = sentence[tok.i:next_tok.i + 1]
                    return span
    return None

def get_wsc_trick_processing(hf_tokenizer):
  
  def simple_encode(text):
    return hf_tokenizer.convert_tokens_to_ids(hf_tokenizer.tokenize(text))

  def wsc_trick_process(sample):
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
    sample['prefix'] = hf_tokenizer.encode(prefix)[:-1] # no SEP
    sample['suffix'] = hf_tokenizer.encode(suffix)[1:] # no CLS
    cands = [simple_encode(sample['span1_text'])] + [simple_encode(cand) for cand in cands]
    sample['cands'] = reduce(operator.add, cands, []) # flatten list, into a 1d token ids
    sample['cand_lens'] = [len(cand) for cand in cands]
    # sample already have 'label'

    return sample

  return wsc_trick_process

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