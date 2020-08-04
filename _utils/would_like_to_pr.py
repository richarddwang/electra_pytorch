from tqdm import tqdm
import sklearn.metrics as skm
import scipy.stats as scs
from torch import nn
from fastai2.text.all import *

"""
There is bug in scikit_learn (https://github.com/scikit-learn/scikit-learn/issues/16924)
so you always get `RuntimeWarning: invalid value encountered in double_scalar` and get a value 0.0
"""
def MatthewsCorrCoef(sample_weight=None, **kwargs):
    return skm_to_fastai(skm.matthews_corrcoef, sample_weight=sample_weight, **kwargs)

"""
If you see PearsonRConstantInputWarning, that may mean the model always outputs a specific label,
which is probable when you test something with very small dataset size or very short training
"""
@delegates(AccumMetric.__init__)
def scs_to_fastai(func, dim_argmax=-1, **kwargs):
  return AccumMetric(func, dim_argmax=dim_argmax, **kwargs)

@delegates(scs_to_fastai)
def PearsonCorrCoef(**kwargs):
    "Pearson correlation coefficient for regression problem"
    def pearsonr(x,y): return scs.pearsonr(x,y)[0]
    return scs_to_fastai(pearsonr, invert_arg=False, dim_argmax=None, **kwargs)
"""
For the same reason as Pearson correlation, you may see nan for value of Spearman correlation, and
RuntimeWarning: invalid value encountered in true_divide, because there is no mutual change. 
(see https://stackoverflow.com/questions/45897003/python-numpy-corrcoef-runtimewarning-invalid-value-encountered-in-true-divide)
"""
# metric for STS task
@delegates(scs_to_fastai)
def SpearmanCorrCoef(axis=0, nan_policy='propagate', **kwargs):
    "Spearman correlation coefficient for regression problem"
    def spearmanr(a,b=None,**kwargs): return scs.spearmanr(a,b,**kwargs)[0]
    return scs_to_fastai(spearmanr, invert_arg=False, dim_argmax=None, axis=axis, nan_policy=nan_policy, **kwargs)

"""
I would like more uniform way to pass the metrics, no matter loss_func or metric,
instantiate it and then pass.
This uniform way also make it possible such as `metrics=[m() for m inTASK_METRICS[task]]`
"""
def Accuracy(axis=-1):
  return AvgMetric(partial(accuracy, axis=axis))


@log_args
@delegates(keep=True)
class LabelSmoothingCrossEntropyFlat(BaseLoss):
    "Same as `nn.CrossEntropyLoss`, but flattens input and target."
    y_int = True
    def __init__(self, *args, axis=-1, **kwargs): super().__init__(LabelSmoothingCrossEntropy, *args, axis=axis, **kwargs)
    def activation(self, out): return F.softmax(out, dim=-1)
    def decodes(self, out):    return out.argmax(dim=-1)

@delegates()
class MyMSELossFlat(BaseLoss):
  def __init__(self,*args, axis=-1, floatify=True, low=None, high=None, **kwargs):
    super().__init__(nn.MSELoss, *args, axis=axis, floatify=floatify, is_2d=False, **kwargs)
    self.low, self.high = low, high
  def decodes(self, x):
    if self.low is not None: x = torch.max(x, x.new_full(x.shape, self.low))
    if self.high is not None: x = torch.min(x, x.new_full(x.shape, self.high))
    return x

my_norm_types = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.InstanceNorm1d, nn.InstanceNorm2d, nn.InstanceNorm3d, nn.LayerNorm)

# Cell
def my_bn_bias_params(m, with_bias=True): # TODO: Rename to `norm_bias_params`
    "Return all bias and BatchNorm parameters"
    if isinstance(m, my_norm_types): return L(m.parameters())
    res = L(m.children()).map(my_bn_bias_params, with_bias=with_bias).concat()
    if with_bias and getattr(m, 'bias', None) is not None: res.append(m.bias)
    return res

def my_bn_bias_state(self, with_bias): return my_bn_bias_params(self.model, with_bias).map(self.opt.state)