import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
from torch.autograd import Variable

def get_param_pytorch(orig_list,model):
    for k,v in module.state_dict().items():
        orig_list.append(v.cpu().numpy())
    return orig_list

def set_params_from_np_list(params_list, model, device):
    params_list = [torch.from_numpy(x) for x in params_list];
    model.conv1.weight.data = params_list[0]
    model.conv1.bias.data = params_list[2]
    model.conv2.weight.data = params_list[1]
    model.conv2.bias.data = params_list[3]
    model.fc1.weight.data = params_list[4]
    model.fc1.bias.data params_list[5]
    model.fc2.weight.data = params_list[6]
    model.fc2.bias.data = paramas_list[7]
    
    model.to(device)

'''need to finish the function'''
def get_grad(model, data, target, loss, device):
    data, target = data.to(device), target.to(device)
    optimizer.zero_grad()
    loss = criterion(output, target)
    loss.backward()


# write the scoring function to decide which submodel to train
# weight_after and weights_before are form of list of numpy
def scoring_weightchange(weights_before, weights_after):
    # define the score = |weight_after_training - weight_before_training|/weight_before_traing
    # score = |weight_after_training - weight_before_training|
    weights_after = weights_after
    weigths_before = weights_before
    score = []
    for k in range(len(weights_after)):
        top = np.absolute(weights_before[k] - weights_after[k])
        bottom  = weights_before[k]
        score.append(np.divide(top,bottom))
    return score


def scoring_grad(model, grad):
    # define the score depending on the magnitude of grad
    # get grad in the form of list of numpy
    score = []
    for k in range(len(grad)):
        top = np.absolute(grad[k])
        '''  maybe we should normalise the score by sum of grad in the same level '''
        bottom = np.sum(grad[k])
        score.append(top/bottom)


def sendback_mask(scores, threshold):
    # given scores and threshold, define the sendback mask, and only send back the submodel
    scores_biglist = [x for sublist in scores for x in sublist]
    thres = np.percentile(scores_biglist,threshold)
    for k in range(len(scores)):
        w = scores[k]- thres 
        w[w>=0] = 1
        w[w<0] = 0


def dropout_mask(scores, threshold):
    # given the scores and threshold, define the dropout mask, and only train the submodel on clients
    scores_biglist = [x for sublist in scores for x in sublist]
    thres = np.percentile(scores_biglist,threshold)
    for k in range(len(scores)):
        w = scores[k]- thres 
        w[w>=0] = 1
        w[w<0] = 0
