# vanilla: 5633, 5633, 5633
# Does "" refer to "" ? :625, 365, 634
# "" = "" ? 6346 6346 6346
#  = ? 6346 5865 6346
#  = ? & discriminator output layer 3653 4807 3653
# A = B and B = A 6346 6346 6346
# A = B and B = A, valid random. 6346 6346 6346
# A=B[sep] or [sep]B=A 6346 6442 6346
# cand 6346 6346 6346 # 6346 from firts epoch
# label smooth 6346 6346 6346 # validation loss drop from to 7x or 8x

# large test
# dev 67 -> test 66.4

if c.wsc_ijiri:
  from _utils.wsc_trick import *
  equal_token_id = hf_tokenizer.convert_tokens_to_ids(['='])[0]
  def wsc_preprocess(examples, augment=False):
    new_examples = {'inp_ids':[], 'attn_mask':[], 'token_type_ids':[], 'label':[]}
    for i in range(len(examples['span1_text'])):
      # clean up the candidate
      the_candidate = examples['span1_text'][i]
      the_candidate = the_candidate.replace('\n',' ')
      if the_candidate.endswith('.') or the_candidate.endswith(','):
        the_candidate = the_candidate[:-1]
      # clean up pronoun
      pronoun = examples['span2_text'][i].rstrip('.,"')
      # find other pronounce
      if augment:
        cand_spans = filter_noun_chunks(
                  extended_noun_chunks(spnlp(examples['text'][i])),
                  exclude_pronouns=True,
                  exclude_query=the_candidate,
                  exact_match=False,
              )
        other_candidates = [str(span) for span in cand_spans]
      else:
        other_candidates = []

      for candidate, label in zip([the_candidate, *other_candidates], [examples['label'][i]] + [0]*len(other_candidates)):
        # random choose
        select = random.randint(0,1)
        # 1
        if select==0 or augment:
          query = f'{pronoun} = {candidate} ?'
          sample = hf_tokenizer.encode_plus(query, examples['text'][i])
          new_examples['inp_ids'].append(sample['input_ids'])
          new_examples['attn_mask'].append(sample['attention_mask'])
          new_examples['token_type_ids'].append(sample['token_type_ids'])
          new_examples['label'].append(label)
        # 2
        if select==1 or augment:
          query = f'{candidate} = {pronoun} ?'
          sample = hf_tokenizer.encode_plus(examples['text'][i], query)
          new_examples['inp_ids'].append(sample['input_ids'])
          new_examples['attn_mask'].append(sample['attention_mask'])
          new_examples['token_type_ids'].append(sample['token_type_ids'])
          new_examples['label'].append(label)
    return new_examples

  wsc = datasets.load_dataset('super_glue', 'wsc', cache_dir='./datasets')
  #bk()
  glue_dsets['wnli'] = wsc.my_map(wsc_preprocess, batched=True, 
                                  remove_columns=wsc['train'].column_names,
                                  fn_kwargs={'train':{'augment': True}},
                                  cache_file_names="ijiri_{split}.arrow")
  hf_dsets = HF_Datasets(glue_dsets['wnli'], hf_toker=hf_tokenizer, n_inp=3,
                cols={'inp_ids':TensorText, 'attn_mask':noop, 'token_type_ids':noop, 'label':TensorCategory})
  glue_dls['wnli'] = hf_dsets.dataloaders(bs=32, shuffle_train=True, num_workers=c.num_workers,
                                        cache_name="dl_ijiri_{split}.json")
  LOSS_FUNC['wnli'] = LabelSmoothingCrossEntropyFlat()

# if c.wsc_ijiri:
#   equal_token_id = hf_tokenizer.convert_tokens_to_ids(['='])[0]
#   def wsc_preprocess(example):
#     # clean up the candidate
#     candidate = example['span1_text']
#     candidate = candidate.replace('\n',' ')
#     if candidate.endswith('.') or candidate.endswith(','):
#       candidate = candidate[:-1]
#     # clean up pronoun
#     pronoun = example['span2_text'].rstrip('.,"')
#     # create the query
#     query = f'{pronoun} = {candidate} ?'
#     # tokenizer
#     sample = hf_tokenizer.encode_plus(example['text'], query)
#     # search for pronoun span
#     start_idx = -(sample['input_ids'][::-1].index(equal_token_id)-1)
#     assert start_idx < -1
#     # create example
#     example['inp_ids'] = sample['input_ids']
#     example['attn_mask'] = sample['attention_mask']
#     example['token_type_ids'] = sample['token_type_ids']
#     example['pronoun_span'] = [start_idx, -1]
#     return example
#   wsc = datasets.load_dataset('super_glue', 'wsc', cache_dir='./datasets')
#   glue_dsets['wnli'] = wsc.my_map(wsc_preprocess,
#                                   cache_file_names="ijiri_{split}.arrow")
#   hf_dsets = HF_Datasets(glue_dsets['wnli'], hf_toker=hf_tokenizer, n_inp=4,
#                 cols={'inp_ids':TensorText, 'attn_mask':noop, 'token_type_ids':noop, 'pronoun_span':noop, 'label':TensorCategory})
#   glue_dls['wnli'] = hf_dsets.dataloaders(bs=32, shuffle_train=True, num_workers=c.num_workers,
#                                         cache_name="dl_ijiri_{split}.json")
#   class IjiriModel(nn.Module):
#     def __init__(self, discriminator):
#       super().__init__()
#       self.discriminator = discriminator
#     def forward(self, input_ids, attention_mask, token_type_ids, pronoun_span):
#       logits = self.discriminator(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)[0] # (B,L)
#       scores = []
#       for i, (start,end) in enumerate(pronoun_span.tolist()):
#         scores.append(logits[i,start:end].mean())
#       return torch.stack(scores).float()

#   class IjiriLoss():
#     def __init__(self):
#       self.criterion = nn.BCEWithLogitsLoss()
#     def __call__(self, scores, y):
#       return self.criterion(scores.float(), y.float())
#     def decodes(self, scores):
#       return scores <= 0
      
#   def wsc_ijiri_accuracy(scores, targs):
#     predicts = scores <= 0
#     return (predicts == targs).float().mean()

#   LOSS_FUNC['wnli'] = IjiriLoss()
#   METRICS['wnli'] = [wsc_ijiri_accuracy]

# elif task == 'wnli' and c.wsc_ijiri:
#     model = IjiriModel(discriminator)

# or (task == 'wnli' and c.wsc_ijiri)