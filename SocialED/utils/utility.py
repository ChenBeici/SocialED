# -*- coding: utf-8 -*-
"""A set of utility functions to support social event detection tasks."""


import torch
import torch.nn.functional as F
import numpy as np
import os
from sklearn.cluster import KMeans
from sentence_transformers import SentenceTransformer
from sklearn.metrics.cluster import normalized_mutual_info_score, adjusted_mutual_info_score, adjusted_rand_score
import warnings
from importlib import import_module
import torch.nn as nn
from itertools import combinations
from scipy import sparse
from torch.utils.data import Dataset
import pandas as pd
from datetime import datetime
import networkx as nx
import dgl
import re
import numpy as np
from .metrics.metric import *



def tokenize_text(text, max_length=512):
    """Tokenize text for social event detection tasks.
    
    Parameters
    ----------
    text : str
        The input text to tokenize.
    max_length : int, optional (default=512)
        Maximum length of tokenized sequence.
        
    Returns
    -------
    tokens : list
        List of tokenized words/subwords.
    """
    # Remove extra whitespace
    text = ' '.join(text.split())
    
    # Basic tokenization by splitting on whitespace
    tokens = text.lower().split()
    
    # Truncate if exceeds max length
    if len(tokens) > max_length:
        tokens = tokens[:max_length]
        
    return tokens


def pprint(params, offset=0, printer=repr):
    """Pretty print the dictionary 'params'.
    
    Parameters
    ----------
    params : dict
        The dictionary to pretty print
    offset : int, optional (default=0)
        The offset at the beginning of each line
    printer : callable, optional (default=repr)
        The function to convert entries to strings
        
    Returns
    -------
    str
        Pretty printed string representation
    """

    params_list = list()
    this_line_length = offset
    line_sep = ',\n' + (1 + offset) * ' '
    for i, (k, v) in enumerate(sorted(params.items())):
        if type(v) is float:
            # use str for representing floating point numbers
            # this way we get consistent representation across
            # architectures and versions.
            this_repr = '%s=%s' % (k, str(v))
        else:
            # use repr of the rest
            this_repr = '%s=%s' % (k, printer(v))
        if len(this_repr) > 500:
            this_repr = this_repr[:300] + '...' + this_repr[-100:]
        if i > 0:
            if this_line_length + len(this_repr) >= 75 or '\n' in this_repr:
                params_list.append(line_sep)
                this_line_length = len(line_sep)
            else:
                params_list.append(', ')
                this_line_length += 2
        params_list.append(this_repr)
        this_line_length += len(this_repr)

    lines = ''.join(params_list)
    # Strip trailing space to avoid nightmare in doctests
    lines = '\n'.join(l.rstrip(' ') for l in lines.split('\n'))
    return lines



def validate_device(gpu_id):
    """Validate the input GPU ID is valid on the given environment.
    If no GPU is presented, return 'cpu'.

    Parameters
    ----------
    gpu_id : int
        GPU ID to check.

    Returns
    -------
    device : str
        Valid device, e.g., 'cuda:0' or 'cpu'.
    """

    # cast to int for checking
    gpu_id = int(gpu_id)

    # if it is cpu
    if gpu_id == -1:
        return 'cpu'

    # if gpu is available
    if torch.cuda.is_available():
        # check if gpu id is between 0 and the total number of GPUs
        check_parameter(gpu_id, 0, torch.cuda.device_count(),
                        param_name='gpu id', include_left=True,
                        include_right=False)
        device = 'cuda:{}'.format(gpu_id)
    else:
        if gpu_id != 'cpu':
            warnings.warn('The cuda is not available. Set to cpu.')
        device = 'cpu'

    return device



def check_parameter(value, lower, upper, param_name, include_left=True, include_right=True):
    """Check if a parameter value is within specified bounds.

    Parameters
    ----------
    value : int or float
        The parameter value to check
    lower : int or float 
        Lower bound
    upper : int or float
        Upper bound
    param_name : str
        Name of the parameter for error messages
    include_left : bool, optional (default=True)
        Whether to include lower bound in valid range
    include_right : bool, optional (default=True)
        Whether to include upper bound in valid range

    Returns
    -------
    bool
        True if parameter is valid, raises ValueError otherwise
    """

    if include_left:
        if value < lower:
            raise ValueError(f"{param_name} must be greater than or equal to {lower}")
    if include_right:
        if value > upper:
            raise ValueError(f"{param_name} must be less than or equal to {upper}")
    return True


def generateMasks(length, data_split, train_i, i, validation_percent=0.1, test_percent=0.2, save_path=None):
    """Generate train/validation/test masks for splitting data.

    Parameters
    ----------
    length : int
        Total number of samples
    data_split : list
        List containing number of samples in each split
    train_i : int
        Index of training split
    i : int
        Current split index
    validation_percent : float, optional (default=0.1)
        Percentage of data to use for validation
    test_percent : float, optional (default=0.2) 
        Percentage of data to use for testing
    save_path : str, optional (default=None)
        Path to save the generated masks

    Returns
    -------
    train_indices : torch.Tensor
        Indices for training set
    validation_indices : torch.Tensor
        Indices for validation set  
    test_indices : torch.Tensor
        Indices for test set
    """
    # verify total number of nodes
    assert length == data_split[i]
    if train_i == i:
        # randomly shuffle the graph indices
        train_indices = torch.randperm(length)
        # get total number of validation indices
        n_validation_samples = int(length * validation_percent)
        # sample n_validation_samples validation indices and use the rest as training indices
        validation_indices = train_indices[:n_validation_samples]
        n_test_samples = n_validation_samples + int(length * test_percent)
        test_indices = train_indices[n_validation_samples:n_test_samples]
        train_indices = train_indices[n_test_samples:]

        if save_path is not None:
            torch.save(validation_indices, save_path + '/validation_indices.pt')
            torch.save(train_indices, save_path + '/train_indices.pt')
            torch.save(test_indices, save_path + '/test_indices.pt')
            validation_indices = torch.load(save_path + '/validation_indices.pt')
            train_indices = torch.load(save_path + '/train_indices.pt')
            test_indices = torch.load(save_path + '/test_indices.pt')
        return train_indices, validation_indices, test_indices
    # If is in inference(prediction) epochs, generate test indices
    else:
        test_indices = torch.range(0, (data_split[i] - 1), dtype=torch.long)
        if save_path is not None:
            torch.save(test_indices, save_path + '/test_indices.pt')
            test_indices = torch.load(save_path + '/test_indices.pt')
        return test_indices



def currentTime():
    """Get current time as formatted string.
    
    Returns
    -------
    str
        Current time in format 'YYYY-MM-DD HH:MM:SS'
    """
    return datetime.now().strftime('%Y-%m-%d %H:%M:%S')


class EMA:  #Exponential Moving Average
    def __init__(self, beta, epochs):
        super().__init__()
        self.beta = beta
        self.step = 0
        self.total_steps = epochs

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new

def get_task(strs):
    tasks = ["DRL","random","semi-supervised","traditional"]
    if len(strs) == 1:
        return "DRL"
    if ("--task" in strs) and len(strs) == 2:
        return "DRL"
    if ("--task" not in strs) or len(strs)!=3:
        return False
    elif strs[-1] not in tasks:
        return False
    else:
        return strs[-1]

def init_weights(m):
    """Initialize model weights using Xavier initialization.
    
    Parameters
    ----------
    m : torch.nn.Module
        Neural network module to initialize
    """
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)
        
def sim(z1, z2):
    """Compute cosine similarity between two sets of vectors.
    
    Parameters
    ----------
    z1 : torch.Tensor
        First set of vectors
    z2 : torch.Tensor
        Second set of vectors
        
    Returns
    -------
    torch.Tensor
        Similarity matrix
    """
    z1 = F.normalize(z1)
    z2 = F.normalize(z2)
    return torch.mm(z1, z2.t())

def intersection(lst1, lst2):
    lst3 = [value for value in lst1 if value in lst2]
    return lst3


def relu_evidence(y):
    return F.relu(y)

def exp_evidence(y):
    return torch.exp(torch.clamp(y, -10, 10))

def softplus_evidence(y):
    return F.softplus(y)

def kl_divergence(alpha, num_classes, device):
    ones = torch.ones([1, num_classes], dtype=torch.float32, device=device)
    sum_alpha = torch.sum(alpha, dim=1, keepdim=True)
    first_term = (
        torch.lgamma(sum_alpha)
        - torch.lgamma(alpha).sum(dim=1, keepdim=True)
        + torch.lgamma(ones).sum(dim=1, keepdim=True)
        - torch.lgamma(ones.sum(dim=1, keepdim=True))
    )
    second_term = (
        (alpha - ones)
        .mul(torch.digamma(alpha) - torch.digamma(sum_alpha))
        .sum(dim=1, keepdim=True)
    )
    kl = first_term + second_term
    return kl

def pairwise_sample(embeddings, labels=None, model=None):
    if model == None:
        labels = labels.cpu().data.numpy()
        indices = np.arange(0,len(labels),1)
        pairs = np.array(list(combinations(indices, 2)))
        pair_labels = (labels[pairs[:,0]]==labels[pairs[:,1]])

        pair_matrix = np.eye(len(labels))
        ind = np.where(pair_labels)
        pair_matrix[pairs[ind[0],0],pairs[ind[0],1]] = 1
        pair_matrix[pairs[ind[0],1], pairs[ind[0],0]] = 1

        return torch.LongTensor(pairs), torch.LongTensor(pair_labels.astype(int)),torch.LongTensor(pair_matrix)

    else:
        pair_matrix = model(embeddings)
        return pair_matrix

def replaceAtUser(text):
    """ Replaces "@user" with "" """
    text = re.sub('@[^\s]+|RT @[^\s]+','',text)
    return text

def removeUnicode(text):
    """ Removes unicode strings like "\u002c" and "x96" """
    text = re.sub(r'(\\u[0-9A-Fa-f]+)',r'', text)       
    text = re.sub(r'[^\x00-\x7f]',r'',text)
    return text

def replaceURL(text):
    """ Replaces url address with "url" """
    text = re.sub('((www\.[^\s]+)|(https?://[^\s]+))','url',text)
    text = re.sub(r'#([^\s]+)', r'\1', text)
    return text

def replaceMultiExclamationMark(text):
    """ Replaces repetitions of exlamation marks """
    text = re.sub(r"(\!)\1+", '!', text)
    return text

def replaceMultiQuestionMark(text):
    """ Replaces repetitions of question marks """
    text = re.sub(r"(\?)\1+", '?', text)
    return text

def removeEmoticons(text):
    """ Removes emoticons from text """
    text = re.sub(':\)|;\)|:-\)|\(-:|:-D|=D|:P|xD|X-p|\^\^|:-*|\^\.\^|\^\-\^|\^\_\^|\,-\)|\)-:|:\'\(|:\(|:-\(|:\S|T\.T|\.\_\.|:<|:-\S|:-<|\*\-\*|:O|=O|=\-O|O\.o|XO|O\_O|:-\@|=/|:/|X\-\(|>\.<|>=\(|D:', '', text)
    return text

def removeNewLines(text):
    text = re.sub('\n', '', text)
    return text

def preprocess_sentence(s):
    return removeNewLines(replaceAtUser(removeEmoticons(replaceMultiQuestionMark(replaceMultiExclamationMark(removeUnicode(replaceURL(s)))))))

def preprocess_french_sentence(s):
    return removeNewLines(replaceAtUser(removeEmoticons(replaceMultiQuestionMark(replaceMultiExclamationMark(replaceURL(s))))))

def SBERT_embed(s_list, language = 'English'):
    '''
    Use Sentence-BERT to embed sentences.
    s_list: a list of sentences/ tokens to be embedded.
    output: the embeddings of the sentences/ tokens.
    '''
    if language == 'English':
        model = SentenceTransformer('all-MiniLM-L6-v2') # for English
    elif language == 'French':
        import os
        model = SentenceTransformer('SBERT',trust_remote_code=True) # for French:distiluse-base-multilingual-cased-v1
    embeddings = model.encode(s_list, convert_to_tensor = True, normalize_embeddings = True)
    return embeddings.cpu()

def evaluate_metrics(labels_true, labels_pred):
    nmi = normalized_mutual_info_score(labels_true, labels_pred)
    ami = adjusted_mutual_info_score(labels_true, labels_pred)
    ari = adjusted_rand_score(labels_true, labels_pred)
    return nmi, ami, ari

def decode(division):
    if type(division) is dict:
        prediction_dict = {m: event for event, messages in division.items() for m in messages}
    elif type(division) is list:
        prediction_dict = {m: event for event, messages in enumerate(division) for m in messages}
    prediction_dict_sorted = dict(sorted(prediction_dict.items()))
    return list(prediction_dict_sorted.values())

def make_onehot(input, classes):
    input = torch.LongTensor(input).unsqueeze(1)
    result = torch.zeros(len(input),classes).long()
    result.scatter_(dim=1,index=input.long(),src=torch.ones(len(input),classes).long())
    return result

def DS_Combin(alpha, classes):
    """
    :param alpha: All Dirichlet distribution parameters.
    :return: Combined Dirichlet distribution parameters.
    """

    def DS_Combin_two(alpha1, alpha2, classes):
        """
        :param alpha1: Dirichlet distribution parameters of view 1
        :param alpha2: Dirichlet distribution parameters of view 2
        :return: Combined Dirichlet distribution parameters
        """
        alpha = dict()
        alpha[0], alpha[1] = alpha1, alpha2
        b, S, E, u = dict(), dict(), dict(), dict()
        for v in range(2):
            S[v] = torch.sum(alpha[v], dim=1, keepdim=True)
            E[v] = alpha[v] - 1
            b[v] = E[v] / (S[v].expand(E[v].shape))
            u[v] = classes / S[v]

        # b^0 @ b^(0+1)
        bb = torch.bmm(b[0].view(-1, classes, 1), b[1].view(-1, 1, classes))
        # b^0 * u^1
        uv1_expand = u[1].expand(b[0].shape)
        bu = torch.mul(b[0], uv1_expand)
        # b^1 * u^0
        uv_expand = u[0].expand(b[0].shape)
        ub = torch.mul(b[1], uv_expand)
        # calculate C
        bb_sum = torch.sum(bb, dim=(1, 2), out=None)
        bb_diag = torch.diagonal(bb, dim1=-2, dim2=-1).sum(-1)
        C = bb_sum - bb_diag

        # calculate b^a
        b_a = (torch.mul(b[0], b[1]) + bu + ub) / ((1 - C).view(-1, 1).expand(b[0].shape))
        # calculate u^a
        u_a = torch.mul(u[0], u[1]) / ((1 - C).view(-1, 1).expand(u[0].shape))

        # calculate new S
        S_a = classes / u_a
        # calculate new e_k
        e_a = torch.mul(b_a, S_a.expand(b_a.shape))
        alpha_a = e_a + 1
        return alpha_a, u_a

    if len(alpha)==1:
        S = torch.sum(alpha[0], dim=1, keepdim=True)
        u = classes / S
        return alpha[0],u
    for v in range(len(alpha) - 1):
        if v == 0:
            alpha_a,u_a = DS_Combin_two(alpha[0], alpha[1], classes)
        else:
            alpha_a,u_a = DS_Combin_two(alpha_a, alpha[v + 1], classes)
    return alpha_a,u_a



def graph_statistics(G, save_path):
    message = '\nGraph statistics:\n'
    num_nodes = G.number_of_nodes()
    num_edges = G.number_of_edges()
    ave_degree = (num_edges / 2) // num_nodes
    in_degrees = G.in_degrees()
    isolated_nodes = torch.zeros([in_degrees.size()[0]], dtype=torch.long)
    isolated_nodes = (in_degrees == isolated_nodes)
    torch.save(isolated_nodes, save_path + '/isolated_nodes.pt')
    num_isolated_nodes = torch.sum(isolated_nodes).item()
    message += 'We have ' + str(num_nodes) + ' nodes.\n'
    message += 'We have ' + str(num_edges / 2) + ' in-edges.\n'
    message += 'Average degree: ' + str(ave_degree) + '\n'
    message += 'Number of isolated nodes: ' + str(num_isolated_nodes) + '\n'
    print(message)
    with open(save_path + "/graph_statistics.txt", "w") as f:
        f.write(message)
    return num_isolated_nodes


def get_dgl_data(dataset, views):
    g_dict = {}
    path = "../data/{}/".format(dataset)
    features = torch.FloatTensor(np.load(path + "features.npy"))
    times = np.load(path + "time.npy")
    times = torch.FloatTensor(((times - times.min()).astype('timedelta64[D]') / np.timedelta64(1, 'D')))
    labels = np.load(path + "label.npy")
    for v in views:
        if v == "h":
            matrix = sparse.load_npz(path + "s_tweet_tweet_matrix_{}.npz".format(v))
        else:
            matrix = sparse.load_npz(path+"s_tweet_tweet_matrix_{}.npz".format(v))
        g = dgl.DGLGraph(matrix, readonly=True)
        save_path_v = path + v
        if not os.path.exists(save_path_v):
            os.mkdir(save_path_v)
        num_isolated_nodes = graph_statistics(g, save_path_v)
        g.set_n_initializer(dgl.init.zero_initializer)
        g.readonly(readonly_state=True)
        g_dict[v] = g
    return g_dict, times, features, labels

def ava_split_data(length, labels, classes):
    indices = torch.randperm(length)
    labels = torch.LongTensor(labels[indices])

    train_indices = []
    test_indices = []
    val_indices = []

    for l in range(classes):
        l_indices = torch.LongTensor(np.where(labels.numpy() == l)[0].reshape(-1))
        val_indices.append(l_indices[:20].reshape(-1,1))
        test_indices.append(l_indices[20:50].reshape(-1,1))
        train_indices.append(l_indices[50:].reshape(-1,1))

    val_indices = indices[torch.cat(val_indices,dim=0).reshape(-1)]
    test_indices = indices[torch.cat(test_indices,dim=0).reshape(-1)]
    train_indices = indices[torch.cat(train_indices,dim=0).reshape(-1)]
    print(train_indices.shape,val_indices.shape,test_indices.shape)
    print(train_indices)
    return train_indices, val_indices, test_indices


def semi_loss(z1, z2):
    f = lambda x: torch.exp(x / 0.05)
    refl_sim = f(sim(z1, z1))
    between_sim = f(sim(z1, z2))

    return -torch.log(between_sim.diag() / (refl_sim.sum(1) + between_sim.sum(1) - refl_sim.diag()))

def get_loss(h1, h2):
    l1 = semi_loss(h1, h2)
    l2 = semi_loss(h2, h1)

    ret = (l1 + l2) * 0.5
    ret = ret.mean()

    return ret

def update_moving_average(ema_updater, ma_model, current_model):
    for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
        old_weight, up_weight = ma_params.data, current_params.data
        ma_params.data = ema_updater.update_average(old_weight, up_weight)


def set_requires_grad(model, val):
    #set require_grad
    for p in model.parameters():
        p.requires_grad = val

def enumerateConfig(args):
    args_names = []
    args_vals = []
    for arg in vars(args):
        args_names.append(arg)
        args_vals.append(getattr(args, arg))

    return args_names, args_vals


def config2string(args):
    args_names, args_vals = enumerateConfig(args)
    st = ''
    for name, val in zip(args_names, args_vals):
        if val == False:
            continue
        if name not in ['device','root','epochs','isAnneal','dropout','warmup_step','clus_num_iters']:
            st_ = "{}_{}_".format(name, val)
            st += st_

    return st[:-1]


def printConfig(args): 
    args_names, args_vals = enumerateConfig(args)
    print(args_names)
    print(args_vals)



