"""
We use simplebooks (https://arxiv.org/abs/1911.12391) as our demo pretrainig corpus. Note that used here is clean version by me for my personal use. The clean version discard chapter and book names, and try to split too long sentence into multiple sentences... .
"""

import pandas as pd
from pathlib import Path

def load_demo_dataframe():
  with Path('./_utils/simplebooks-2_clean_valid.txt').open() as f: v_text = f.read()
  with Path('./_utils/simplebooks-2_clean_test.txt').open() as f: t_text = f.read()
  v_sents, t_sents = v_text.split('\n\n'), t_text.split('\n\n')
  df = pd.DataFrame.from_dict({
    'text': v_sents + t_sents,
    'is_valid': [False]*len(v_sents) + [True]*len(t_sents),
  })
  return df  
