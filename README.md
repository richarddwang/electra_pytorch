Unofficial PyTorch implementation of 

> [ELECTRA: Pre-training Text Encoders as Discriminators Rather Than Generators](https://arxiv.org/abs/2003.10555)
> by Kevin Clark. Minh-Thang Luong. Quoc V. Le. Christopher D. Manning

# Replicated Results
I pretrain ELECTRA-small from scratch and have successfully replicated the paper's results on GLUE. 

|Model|CoLA|SST|MRPC|STS|QQP|MNLI|QNLI|RTE|Avg. of Avg.|
|---|---|---|---|---|---|---|---|---|---|
|ELECTRA-Small-OWT|56.8|88.3|87.4|86.8|88.3|78.9|87.9|68.5|80.36|
|ELECTRA-Small-OWT (my)|58.72|88.03|86.04|86.16|88.63|80.4|87.45|67.46|80.36

*Table 1:* Results on GLUE dev set. The official result comes from [expected results](https://github.com/google-research/electra#expected-results). Scores are the average scores finetuned from the same checkpoint. (See [this issue](https://github.com/google-research/electra/issues/98)) My result comes from pretraining a model from scratch and thens taking average from 10 finetuning runs for each task. Both results are trained on OpenWebText corpus


|Model|CoLA|SST|MRPC|STS|QQP|MNLI|QNLI|RTE|Avg.|
|---|---|---|---|---|---|---|---|---|---|
|ELECTRA-Small++|55.6|91.1|84.9|84.6|88.0|81.6|88.3|6.36|79.7|
|ELECTRA-Small++ (my)|54.8|91.6|84.6|84.2|88.5|82|89|64.7|79.92

*Table 2:* Results on GLUE test set. My result finetunes the pretrained checkpoint loaded from huggingface.

Official training loss curve |  My training loss curve
:-------------------------:|:-------------------------:
![image](https://user-images.githubusercontent.com/17963619/95172776-65682d80-07ea-11eb-82b1-5fcff9e8d6a8.png)|![image](https://user-images.githubusercontent.com/17963619/95174020-115e4880-07ec-11eb-8d51-baf30a958dfe.png)

*Table 3:* Both are small models trained on OpenWebText. The official one is from [here](https://github.com/google-research/electra/issues/3). You should take the value of training loss with a grain of salt since it doesn't reflect the performance of downstream tasks.

# Features of this implementation

- You don't need to download and process datasets manually, the scirpt take care those for you automatically. (Thanks to [huggingface/datasets](https://github.com/huggingface/datasets) and [hugginface/transformers](https://github.com/huggingface/transformers))

- AFAIK, the closest reimplementation to the original one, taking care of many easily overlooked details (described below). 

- AFAIK, the only one successfully validate itself by replicating the results in the paper.

- Comes with jupyter notebooks, which you can explore the code and inspect the processed data.

- You don't need to download and preprocess anything by yourself, all you need is running the training script.

# More results 
## How stable is ELECTRA pretraining?
|Mean|Std|Max|Min|#models|
|---|---|---|---|---|
|81.38|0.57|82.23|80.42|14|

*Tabel 4:* Statistics of GLUE devset results for small models. Every model is pretrained from scratch with different seeds and finetuned for 10 random runs for each GLUE task. Score of a model is the average of the best of 10 for each task. (The process is as same as the one described in the paper) As we can see, although ELECTRA is mocking adeversarial training, it has a good training stability.

## How stable is ELECTRA finetuing on GLUE ?
|Model|CoLA|SST|MRPC|STS|QQP|MNLI|QNLI|RTE|
|---|---|---|---|---|---|---|---|---|
|ELECTRA-Small-OWT (my)|1.30|0.49|0.7|0.29|0.1|0.15|0.33|1.93

*Table 5:* Standard deviation for each task. This is the same model as Table 1, which finetunes 10 runs for each task.

# Discussion
[HuggingFace forum post](https://discuss.huggingface.co/t/electra-training-reimplementation-and-discussion/1004/7)  
[Fastai forum post ](https://forums.fast.ai/t/electra-training-reimplementation-and-discussion/78280)

# Usage
> Note: This project is actually for my personal research. So I didn't trying to make it easy to use for all users, but trying to make it easy to read and modify.

## Install requirements
`pip3 install -r requirements.txt`

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
Below lists the details of the [original implementation](https://github.com/google-research/electra)/paper that are easy to be overlooked and I have taken care of. I found these details are indispensable to successfully replicate the results of the paper.
## Optimization
- Using Adam optimizer without bias correction (bias correction is default for Adam optimizer in Pytorch and fastai)
- There is a bug of decaying learning rates through layers in the official implementation , so that when finetuing, lr decays more than the stated in the paper. See [_get_layer_lrs](https://github.com/google-research/electra/blob/79111328070e491b287c307906701ebc61091eb2/model/optimization.py#L186). Also see [this issue](https://github.com/google-research/electra/issues/51).
- Using clip gradient
- using 0 weight decay when finetuning on GLUE
- It didn't do warmup and then do linear decay but do them together, which means the learning rate warmups and decays at the same time during the warming up phase. See [here](https://github.com/google-research/electra/blob/79111328070e491b287c307906701ebc61091eb2/model/optimization.py#L36-L43)
## Data processing
- For pretraing data preprocessing, it concatenates and truncates setences to fit the max length, and stops concating when it comes to the end of a document.
- For pretraing data preprocessing, it by chance splits the text into sentence A and sentence B, and also by chance changes the max length
- For finetuning data preprocessing, it follow BERT's way to truncate the longest one of sentence A and B to fit the max length
## Trick
- For MRPC and STS tasks, it augments training data by add the same training data but with swapped sentence A and B. This is called "double_unordered" in the official implementation.
- It didn't mask sentence like BERT, within the mask probability (15% or other value) of tokens,  a token has 85% chance to be replaced with [MASK] and 15% remains the same but no chance to be replaced with a random token.
## Tying parameter
- Input and output word embeddings of generator, and input word embeddings of discriminator. The three are tied together.
- It tie not only word/pos/token type embeddings but also layer norm in the embedding layers of both generator and discriminator.
## Other
- The output layer is initialized by Tensorflow v1's default initialization (i.e. xavier uniform)
- Using gumbel softmax to sample generations from geneartor as input of discriminator
- It use a dropout and a linear layer in the output layer for GLUE finetuning, not what `ElectraClassificationHead` uses.
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
## Original paper
```
@inproceedings{clark2020electra,
  title = {{ELECTRA}: Pre-training Text Encoders as Discriminators Rather Than Generators},
  author = {Kevin Clark and Minh-Thang Luong and Quoc V. Le and Christopher D. Manning},
  booktitle = {ICLR},
  year = {2020},
  url = {https://openreview.net/pdf?id=r1xMH1BtvB}
}
```
## This implementation. 
**I will join RC2020 so maybe there will be another paper for this implementation then. Be sure to check here again when you cite this implementation.**
```
@misc{electra_pytorch,
  author = {Richard Wang},
  title = {PyTorch implementation of ELECTRA},
  year = {2020},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/richarddwang/electra_pytorch}}
}
```
