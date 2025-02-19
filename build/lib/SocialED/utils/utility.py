# -*- coding: utf-8 -*-
"""A set of utility functions to support social event detection tasks."""

import torch
import torch.nn.functional as F
import numpy as np
from sentence_transformers import SentenceTransformer
import warnings
from itertools import combinations
from datetime import datetime
import numpy as np

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

def currentTime():
    """Get current time as formatted string.
    
    Returns
    -------
    str
        Current time in format 'YYYY-MM-DD HH:MM:SS'
    """
    return datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
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

def SBERT_embed(s_list, language):
    '''
    Use Sentence-BERT to embed sentences.
    s_list: a list of sentences/ tokens to be embedded.
    language: the language of the sentences ('English', 'French', 'Arabic').
    output: the embeddings of the sentences/ tokens.
    '''
    # Model paths or names for each language
    model_map = {
        'English': '../model/model_needed/all-MiniLM-L6-v2',
        'French': '../model/model_needed/distiluse-base-multilingual-cased-v1',
        'Arabic': '../model/model_needed/paraphrase-multilingual-mpnet-base-v2'
    }

    # Default model for Hugging Face
    hf_model_map = {
        'English': 'sentence-transformers/all-MiniLM-L6-v2',
        'French': 'sentence-transformers/distiluse-base-multilingual-cased-v1',
        'Arabic': 'sentence-transformers/paraphrase-multilingual-mpnet-base-v2'
    }

    # Print language and model being used
    print(f"Embedding sentences in language: {language}")
    
    # Determine model path
    model_path = model_map.get(language)
    if not model_path:
        raise ValueError(f"Unsupported language: {language}. Supported languages are: {', '.join(model_map.keys())}")

    print(f"Using model: {model_path}")

    # Load the model, downloading if necessary
    try:
        model = SentenceTransformer(model_path)
        print(f"Successfully loaded model from local path: {model_path}")
    except Exception as e:
        print(f"Model {model_path} not found locally. Attempting to download from Hugging Face...")
        model = SentenceTransformer(hf_model_map[language])
        print(f"Model downloaded from Hugging Face: {hf_model_map[language]}")

    # Compute embeddings
    embeddings = model.encode(s_list, convert_to_tensor=True, normalize_embeddings=True)
    print(f"Computed embeddings for {len(s_list)} sentences/tokens.")
    
    return embeddings.cpu()

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


