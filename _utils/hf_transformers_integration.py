"""
I arranged it from ohmeow/blurr(https://github.com/ohmeow/blurr/blob/67359a8f358b9f044ed561401a720ae5715c63cf/blurr/data.py), because it requires py>=3.7 but Colab has py=3.6 and I have simplified it to just for my needs. 

Anyway, checkout ohmeow's fantatistic work !
"""

from fastai2.text.all import *

class HF_Tokenizer():
    """huggingface friendly tokenization function."""
    def __init__(self, hf_tokenizer, **kwargs):
        self.hf_tokenizer = hf_tokenizer
    def __call__(self, items):
        for txt in items: yield self._tokenize(txt)
    def _tokenize(self, txt):
        return [self.hf_tokenizer.cls_token,*self.hf_tokenizer.tokenize(txt), self.hf_tokenizer.sep_token]

class HF_TextBlock(TransformBlock):
    @delegates(Numericalize.__init__)
    def __init__(self, tok_tfm, hf_tokenizer, task=None,
                 hf_batch_tfm=None, vocab=None, max_seq_len=512, **kwargs):
        return super().__init__(type_tfms=[tok_tfm, Numericalize(vocab, **kwargs)],
                                dl_type=SortedDL,
                                dls_kwargs={ 'before_batch': partial(pad_input_chunk, 
                                                                     pad_idx=hf_tokenizer.pad_token_id,
                                                                     pad_first=False) })
    @classmethod
    @delegates(Tokenizer.from_df, keep=True)
    def from_df(cls, text_cols, hf_tokenizer, task=None,
                res_col_name='text', vocab=None, hf_batch_tfm=None, max_seq_len=512, **kwargs):
        """Creates a HF_TextBlock via a pandas DataFrame"""

        # grab hf tokenizer class to do the actual tokenization (via tok_func) and its vocab
        tokenizer_cls = partial(HF_Tokenizer, hf_tokenizer=hf_tokenizer)
        if (vocab is None): vocab = list(hf_tokenizer.get_vocab())

        tok_tfm = Tokenizer.from_df(text_cols,
                                    res_col_name=res_col_name,
                                    tok_func=tokenizer_cls,
                                    rules=[], **kwargs)

        return cls(tok_tfm, hf_tokenizer=hf_tokenizer, task=task,
                   hf_batch_tfm=hf_batch_tfm, vocab=vocab, max_seq_len=max_seq_len)

# To just take hidden features output
class HFModelWrapper(nn.Module):
  def __init__(self,model, pad_id):
    super().__init__()
    self.model = model
    self.pad_id = pad_id
  def forward(self, x):
    attn_mask = x!= self.pad_id
    return self.model(x, attn_mask)[0]
