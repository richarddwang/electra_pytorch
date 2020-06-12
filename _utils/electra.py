from _utils.huggingface import AggregateTransform

class ELECTRADataTransform(AggregateTransform):
  
  def __init__(self, hf_dset, in_col, out_col, max_length, cls_idx, sep_idx):
    self.in_col, self.out_col = in_col, out_col
    self._current_sentences = []
    self._current_length = 0
    self._max_length = max_length
    self._target_length = max_length
    self.cls_idx, self.sep_idx = cls_idx, sep_idx
    super().__init__(hf_dset, inp_cols=[in_col], out_cols=[out_col], 
                    init_attrs=['_current_sentences', '_current_length', '_target_length'])

  # two functions required by AggregateTransform
  def accumulate(self, tokids):
    self.add_line(tokids)
  
  def create_example(self):
    input_ids = self._create_example()
    return {self.out_col: input_ids}

  def add_line(self, tokids):
    """Adds a line of text to the current example being built."""
    self._current_sentences.append(tokids)
    self._current_length += len(tokids)
    if self._current_length >= self._target_length:
      self.commit_example(self.create_example())

  def _create_example(self):
    """Creates a pre-training example from the current list of sentences."""
    # small chance to only have one segment as in classification tasks
    if random.random() < 0.1:
      first_segment_target_length = 100000
    else:
      # -3 due to not yet having [CLS]/[SEP] tokens in the input text
      first_segment_target_length = (self._target_length - 3) // 2

    first_segment = []
    second_segment = []
    for sentence in self._current_sentences:
      # the sentence goes to the first segment if (1) the first segment is
      # empty, (2) the sentence doesn't put the first segment over length or
      # (3) 50% of the time when it does put the first segment over length
      if (len(first_segment) == 0 or
          len(first_segment) + len(sentence) < first_segment_target_length or
          (len(second_segment) == 0 and
           len(first_segment) < first_segment_target_length and
           random.random() < 0.5)):
        first_segment += sentence
      else:
        second_segment += sentence

    # trim to max_length while accounting for not-yet-added [CLS]/[SEP] tokens
    first_segment = first_segment[:self._max_length - 2]
    second_segment = second_segment[:max(0, self._max_length -
                                         len(first_segment) - 3)]

    # prepare to start building the next example
    self._current_sentences = []
    self._current_length = 0
    # small chance for random-length instead of max_length-length example
    if random.random() < 0.05:
      self._target_length = random.randint(5, self._max_length)
    else:
      self._target_length = self._max_length

    return self._make_example(first_segment, second_segment)

  def _make_example(self, first_segment, second_segment):
    """Converts two "segments" of text into a tf.train.Example."""
    input_ids = [self.cls_idx] + first_segment + [self.sep_idx]
    if second_segment:
      input_ids += second_segment + [self.sep_idx]
    return input_ids