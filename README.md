Unofficial PyTorch implementation of 

> [ELECTRA: Pre-training Text Encoders as Discriminators Rather Than Generators](https://arxiv.org/abs/2003.10555)
> by Kevin Clark. Minh-Thang Luong. Quoc V. Le. Christopher D. Manning

This implementation carefully reproduce every bit of the [original implementation](https://github.com/google-research/electra). 

# Replicated Results
I pretrain ELECTRA-small from scratch and has successfully replicate the paper's results on GLUE. 

|Model|CoLA|SST|MRPC|STS|QQP|MNLI|QNLI|RTE|Avg.|
|---|---|---|---|---|---|---|---|---|---|
|ELECTRA-Small|54.6|89.1|83.7|80.3|88.0|79.7|87.7|60.8|78.0|
|ELECTRA-Small (my)|57.2|87.1|82.1|80.4|88|78.9|87.9|63.1|78.08
Results for models on the GLUE test set.

# Features of this implementation

- You don't need to download and process datasets manually, the scirpt take care those for you automatically. (Thanks to [huggingface/nlp](https://github.com/huggingface/nlp) and [hugginface/transformers](https://github.com/huggingface/transformers))

- You can inspect the content of processed data by uncomment or enable `show_batch` in the scripts

- Jupyter notebooks for pretraining and finetuning respectively, where you can explore anything.

- Successfully replicate the paper results for all GLUE tasks.

- Use wikipedia + bookcorpus data

- Take care of easily overlooked details to ensure the quality. See [Advanced Details](#Advanced-Details)

- From data processing to pretraining to finetuning, carefully follow every bit of the original implementation.

# Usage
> Note: This project is actually for my personal research. So I didn't trying to make it easy to use for all users, but trying to make it easy to read and modify.

## Install requirements
`pip install fastai nlp transformers hugdatafast`

## Steps
1. `python pretrain.py`
2. set `pretrained_checkcpoint` in `finetune.py` to use the checkpoint you've pretrained and saved in `electra_pytorch/checkpoints/pretrain`. 
3. `python finetune.py` (with `do_finetune` set to `True`)
4. Go to neptune, pick the best run of 10 runs for each task, and set `th_runs` in `finetune.py` according to the numbers in the names of runs you picked.
5. `python finetune.py` (with `do_finetune` set to `False`), this outpus predictions on testset, you can then compress and send `.tsv`s in `electra_pytorch/test_outputs/<group_name>/*.tsv` to GLUE site to get test score.

## Notes
- I didn't use CLI arguments, so configure options enclosed within `MyConfig` in the python files to your needs before run them. (There're comments below it showing the options for vanilla settings)

- You will need a [Neptune](https://neptune.ai) account and create a neptune project on the website to record GLUE finetuning results. Don't forget to replace `richarddwang/electra-glue` with your neptune project's name

- The python files `pretrian.py`, `finetune.py` are in fact converted from `Pretrain.ipynb` and `Finetune_GLUE.ipynb`. You can also use those notebooks to explore ELECTRA training and finetuning.

# Advanced Details
Below lists the details of the original implementation/paper that are easy to be overlooked, but I have identified and followed. And I found these details are indispensable to successfully replicate the results of the paper.
- Use Adam optimizer **without bias correction** (bias correction is default for Pytorch and fastai Adam optimizer)
- There is a bug in how original implementation decays learning rates through layers. See [_get_layer_lrs](https://github.com/google-research/electra/blob/79111328070e491b287c307906701ebc61091eb2/model/optimization.py#L186)
- Use clip gradient
- For MRPC and STS tasks, it appends the same dataset with swapped sentence1 and sentence2 to the original dataset, and call it "double_unordered"
- For pretraing data preprocessing, it concat and truncate setences to fit the max length, and stop concating when it comes to the end of a document.
- For pretraing data preprocessing, it by chance split the text into sentence A and sentence B, and also by chance change the max length
- For finetuning data preprocessing, it follow BERT's way to truncate the longest one of sentence A and B to fit the max length
- Use gradient clipping
- The output layer is initialized by Tensorflow v1's default initialization which is xavier
- It use gumbel softmax to sample generations from geneartor
- It didn't mask like BERT, but mask for [MASK] for 85% and 15% remains the same
- It didn't do warmup and then do linear decay but do them together, which means the learning rate warmups and decays at the same time when warming up. See [here](https://github.com/google-research/electra/blob/79111328070e491b287c307906701ebc61091eb2/model/optimization.py#L36-L43)
- It use a dropout and a linear layer for GLUE output layer, not what `ElectraClassificationHead` uses.
- It didn't tie input and output embeddings for its generator, which is a common practice applied by many model.
- It tie not only word/pos/token type embeddings but also layer norm in embedding layer, for generator and discriminator.
- All public model of ELECTRA checkpoints are actually ++ model. See [this issue](https://github.com/google-research/electra/issues/68)
- It downscales generator by hidden_size, number of attention heads, and intermediate size, but not number of layers.

# File architecture
If you pretrain, finetune, and generate test results. `electra_pytorch` will generate these for you.
```
project root
|
|── datasets
|   |── glue
|       |── <task>
|       ...
|
|── checkpoints
|   |── pretrain
|   |   |── <base_run_name>_<seed>_<percent>.pth
|   |    ...
|   |
|   |── glue
|       |── <group_name>_<task>_<ith_run>.pth
|       ...
|
|── test_outputs
|   |── <group_name>
|   |   |── CoLA.tsv
|   |   ...
|   | 
|   | ...
```

# Citation
```
@misc{clark2020electra,
    title={ELECTRA: Pre-training Text Encoders as Discriminators Rather Than Generators},
    author={Kevin Clark and Minh-Thang Luong and Quoc V. Le and Christopher D. Manning},
    year={2020},
    eprint={2003.10555},
    archivePrefix={arXiv},
    primaryClass={cs.CL}
}
```