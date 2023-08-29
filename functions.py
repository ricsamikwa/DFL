
import torch
import torch.nn as nn
import torch.nn.init as init
import torchvision
import torchvision.transforms as transforms
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset, Subset


import pickle, struct, socket
from nn_model import *
from configurations import *
import collections
import numpy as np

import logging
logging.basicConfig(level = logging.INFO,format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

np.random.seed(0)
torch.manual_seed(0)

# transform_train = transforms.Compose([
#   transforms.RandomCrop(32, padding=4),
#   transforms.RandomHorizontalFlip(),
#   transforms.ToTensor(),
#   transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
#     ])
# Transformations
RC = transforms.RandomCrop(32, padding=4)
RHF = transforms.RandomHorizontalFlip()
RVF = transforms.RandomVerticalFlip()
NRM = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
TT = transforms.ToTensor()
TPIL = transforms.ToPILImage()

# Transforms object for trainset with augmentation
transform_with_aug = transforms.Compose([TPIL, RC, RHF, TT, NRM])
# Transforms object for testset with NO augmentation
transform_no_aug = transforms.Compose([TT, NRM])

def get_local_dataloader_non_iid(CLIENT_IDEX, cpu_count):
  indices = list(range(N))
  part_tr = indices[int(5000 * 0) : int(5000 * (0+1))]

  
  trainset = torchvision.datasets.CIFAR10(root=dataset_path, train=True, download=True)
  testset = torchvision.datasets.CIFAR10(root=dataset_path, train=False, download=True)
  classDict = {'plane': 0, 'car': 1, 'bird': 2, 'cat': 3, 'deer': 4, 'dog': 5, 'frog': 6, 'horse': 7, 'ship': 8, 'truck': 9}
	
  x_train = trainset.data
  x_test = testset.data
  y_train = trainset.targets
  y_test = testset.targets

  
  # Let's choose cats (class 3 of CIFAR) and dogs (class 5 of CIFAR) as trainset/testset
  cat_dog_trainset = \
    DatasetMaker(
        [get_class_i(x_train, y_train, classDict['cat']),
         get_class_i(x_train, y_train, classDict['dog']),
         get_class_i(x_train, y_train, classDict['horse'])],
        transform_with_aug
    )
    
  subset = Subset(trainset, part_tr)
  trainloader = DataLoader(
		cat_dog_trainset, batch_size=B, shuffle=True, num_workers=cpu_count)

  classes = ('plane', 'car', 'bird', 'cat', 'deer',
		   'dog', 'frog', 'horse', 'ship', 'truck')
  return trainloader,classes


def replace_numbers(input_tensor, mapping, device):
    
    new_tensor = torch.tensor([mapping[int(num)] for num in input_tensor], device=device)
    return new_tensor


def get_test_dataloader(cpu_count=4):
	
  transform_test = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
  testset = torchvision.datasets.CIFAR10(root=dataset_path, train=False, download=True, transform=transform_test)
  testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=cpu_count)

  return testloader

def get_test_dataloader_non_iid(group_id):
  cpu_count=4
  testset = torchvision.datasets.CIFAR10(root=dataset_path, train=False, download=True)
  classDict = {'plane': 0, 'car': 1, 'bird': 2, 'cat': 3, 'deer': 4, 'dog': 5, 'frog': 6, 'horse': 7, 'ship': 8, 'truck': 9}
  
  x_test = testset.data
  y_test = testset.targets

  # Let's choose cats (class 3 of CIFAR) and dogs (class 5 of CIFAR) as trainset/testset
  # cat_dog_testset = \
  # DatasetMaker(
  #       [get_class_i(x_test, y_test, classDict['cat']),
	#        get_class_i(x_test, y_test, classDict['dog']),
  #        get_class_i(x_test, y_test, classDict['horse']),
	#         get_class_i(x_test, y_test, classDict['plane']),
	#        get_class_i(x_test, y_test, classDict['car']),
  #        get_class_i(x_test, y_test, classDict['frog'])
	#       ],
  #       transform_with_aug
  #   )
  # classDict = {'plane': 0, 'car': 1, 'bird': 2, 'cat': 3, 'deer': 4, 'dog': 5, 'frog': 6, 'horse': 7, 'ship': 8, 'truck': 9}


  #### mapping = {0: 0, 1: 1, 2: 6, 3: 8, 4: 9, 5: 8, 6: 9, 7: 7, 8: 2}
  if group_id == 2:
    cat_dog_testset = \
      DatasetMaker(
          [get_class_i(x_test, y_test, classDict['plane']),
          get_class_i(x_test, y_test, classDict['car']),
          get_class_i(x_test, y_test, classDict['frog']),
	        get_class_i(x_test, y_test, classDict['ship']),
		      get_class_i(x_test, y_test, classDict['truck']),
		      get_class_i(x_test, y_test, classDict['cat']),
	        get_class_i(x_test, y_test, classDict['dog']),
		      get_class_i(x_test, y_test, classDict['horse']),
	        get_class_i(x_test, y_test, classDict['bird'])],
          transform_with_aug
      )
    
  #### mapping = {0: 3, 1: 5, 2: 7, 3: 2, 4: 4, 5: 0, 6: 1, 7: 6, 8: 8} 
  if group_id == 1:
    cat_dog_testset = \
      DatasetMaker(
          [get_class_i(x_test, y_test, classDict['cat']),
          get_class_i(x_test, y_test, classDict['dog']),
          get_class_i(x_test, y_test, classDict['horse']),
	        get_class_i(x_test, y_test, classDict['bird']),
          get_class_i(x_test, y_test, classDict['deer']),
          get_class_i(x_test, y_test, classDict['plane']),
	        get_class_i(x_test, y_test, classDict['car']),
		      get_class_i(x_test, y_test, classDict['frog']),
	        get_class_i(x_test, y_test, classDict['ship'])],
          transform_with_aug
      )
  trainloader = DataLoader(
    cat_dog_testset, batch_size=B, shuffle=True, num_workers=cpu_count)

  return trainloader

def get_class_i(x, y, i):
    """
    x: trainset.train_data or testset.test_data
    y: trainset.train_labels or testset.test_labels
    i: class label, a number between 0 to 9
    return: x_i
    """
    # Convert to a numpy array
    y = np.array(y)
    # Locate position of labels that equal to i
    pos_i = np.argwhere(y == i)
    # Convert the result into a 1-D list
    pos_i = list(pos_i[:, 0])
    # Collect all data that match the desired label
    x_i = [x[j] for j in pos_i]

    return x_i

class DatasetMaker(Dataset):
    def __init__(self, datasets, transformFunc=transform_no_aug):
        """
        datasets: a list of get_class_i outputs, i.e. a list of list of images for selected classes
        """
        self.datasets = datasets
        self.lengths = [len(d) for d in self.datasets]
        self.transformFunc = transformFunc

    def __getitem__(self, i):
        class_label, index_wrt_class = self.index_of_which_bin(self.lengths, i)
        img = self.datasets[class_label][index_wrt_class]
        img = self.transformFunc(img)
        return img, class_label

    def __len__(self):
        return sum(self.lengths)

    def index_of_which_bin(self, bin_sizes, absolute_index, verbose=False):
        """
        Given the absolute index, returns which bin it falls in and which element of that bin it corresponds to.
        """
        # Which class/bin does i fall into?
        accum = np.add.accumulate(bin_sizes)
        if verbose:
            print("accum =", accum)
        bin_index = len(np.argwhere(accum <= absolute_index))
        if verbose:
            print("class_label =", bin_index)
        # Which element of the fallent class/bin does i correspond to?
        index_wrt_class = absolute_index - np.insert(accum, 0, 0)[bin_index]
        if verbose:
            print("index_wrt_class =", index_wrt_class)

        return bin_index, index_wrt_class

def get_local_dataloader(CLIENT_IDEX, cpu_count):
	indices = list(range(N))
	part_tr = indices[int(5000 * 0) : int(5000 * (0+1))]

	transform_train = transforms.Compose([
	transforms.RandomCrop(32, padding=4),
	transforms.RandomHorizontalFlip(),
	transforms.ToTensor(),
	transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
	trainset = torchvision.datasets.CIFAR10(
		root=dataset_path, train=True, download=True, transform=transform_train)
	
	subset = Subset(trainset, part_tr)
	trainloader = DataLoader(
		subset, batch_size=B, shuffle=True, num_workers=cpu_count)

	classes = ('plane', 'car', 'bird', 'cat', 'deer',
		   'dog', 'frog', 'horse', 'ship', 'truck')
	return trainloader,classes


def recv_msg(sock, expect_msg_type=None):
	msg_len = struct.unpack(">I", sock.recv(4))[0]
	msg = sock.recv(msg_len, socket.MSG_WAITALL)
	msg = pickle.loads(msg)
	logger.debug(msg[0]+'received from'+str(sock.getpeername()[0])+':'+str(sock.getpeername()[1]))

	if (expect_msg_type is not None) and (msg[0] != expect_msg_type):
		raise Exception("Error: received" + msg[0] + " instead of " + expect_msg_type)
	return msg

def get_model(location, model_name, layer, device, cfg):
	cfg = cfg.copy()
	net = VGG(location, model_name, layer, cfg)
	net = net.to(device)
	logger.debug(str(net))
	return net

def send_msg(sock, msg):
	msg_pickle = pickle.dumps(msg)
	sock.sendall(struct.pack(">I", len(msg_pickle)))
	sock.sendall(msg_pickle)
	logger.debug(msg[0]+'sent to'+str(sock.getpeername()[0])+':'+str(sock.getpeername()[1]))

def split_weights_client(weights,cweights):
	for key in cweights:
		assert cweights[key].size() == weights[key].size()
		cweights[key] = weights[key]
	return cweights

def split_weights_server(weights,cweights,sweights):
	ckeys = list(cweights)
	skeys = list(sweights)
	keys = list(weights)

	for i in range(len(skeys)):
		assert sweights[skeys[i]].size() == weights[keys[i + len(ckeys)]].size()
		sweights[skeys[i]] = weights[keys[i + len(ckeys)]]

	return sweights

def concat_weights(weights,cweights,sweights):
	concat_dict = collections.OrderedDict()

	ckeys = list(cweights)
	skeys = list(sweights)
	keys = list(weights)

	for i in range(len(ckeys)):
		concat_dict[keys[i]] = cweights[ckeys[i]]

	for i in range(len(skeys)):
		concat_dict[keys[i + len(ckeys)]] = sweights[skeys[i]]

	return concat_dict

def zero_init(net):
	for m in net.modules():
		if isinstance(m, nn.Conv2d):
			init.zeros_(m.weight)
			if m.bias is not None:
				init.zeros_(m.bias)
		elif isinstance(m, nn.BatchNorm2d):
			init.zeros_(m.weight)
			init.zeros_(m.bias)
			init.zeros_(m.running_mean)
			init.zeros_(m.running_var)
		elif isinstance(m, nn.Linear):
			init.zeros_(m.weight)
			if m.bias is not None:
				init.zeros_(m.bias)
	return net
def norm_list(alist):	
	return [l / sum(alist) for l in alist]

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
		
def fed_avg(zero_model, w_local_list, totoal_data_size):
	keys = w_local_list[0][0].keys()
	
	for k in keys:
		for w in w_local_list:
			beta = float(w[1]) / float(totoal_data_size)
			if 'num_batches_tracked' in k:
				zero_model[k] = w[0][k]
			else:	
				zero_model[k] += (w[0][k] * beta)

	return zero_model

def model_similarity_cka(features_x, features_y, debiased=False):
		"""Compute CKA with a linear kernel, in feature space.

		This is typically faster than computing the Gram matrix when there are fewer
		features than examples.

		Args:
			features_x: A num_examples x num_features matrix of features.
			features_y: A num_examples x num_features matrix of features.
			debiased: Use unbiased estimator of dot product similarity. CKA may still be
			biased. Note that this estimator may be negative.

		Returns:
			The value of CKA between X and Y.
		"""
		features_x = features_x - np.mean(features_x, 0, keepdims=True)
		features_y = features_y - np.mean(features_y, 0, keepdims=True)

		dot_product_similarity = np.linalg.norm(features_x.T.dot(features_y)) ** 2
		normalization_x = np.linalg.norm(features_x.T.dot(features_x))
		normalization_y = np.linalg.norm(features_y.T.dot(features_y))

		if debiased:
			n = features_x.shape[0]
			# Equivalent to np.sum(features_x ** 2, 1) but avoids an intermediate array.
			sum_squared_rows_x = np.einsum('ij,ij->i', features_x, features_x)
			sum_squared_rows_y = np.einsum('ij,ij->i', features_y, features_y)
			squared_norm_x = np.sum(sum_squared_rows_x)
			squared_norm_y = np.sum(sum_squared_rows_y)

			dot_product_similarity = _debiased_dot_product_similarity_helper(
				dot_product_similarity, sum_squared_rows_x, sum_squared_rows_y,
				squared_norm_x, squared_norm_y, n)
			normalization_x = np.sqrt(_debiased_dot_product_similarity_helper(
				normalization_x ** 2, sum_squared_rows_x, sum_squared_rows_x,
				squared_norm_x, squared_norm_x, n))
			normalization_y = np.sqrt(_debiased_dot_product_similarity_helper(
				normalization_y ** 2, sum_squared_rows_y, sum_squared_rows_y,
				squared_norm_y, squared_norm_y, n))

		return dot_product_similarity / (normalization_x * normalization_y)		
	
def _debiased_dot_product_similarity_helper(
		xty, sum_squared_rows_x, sum_squared_rows_y, squared_norm_x, squared_norm_y,
		n):
	"""Helper for computing debiased dot product similarity (i.e. linear HSIC)."""
	# This formula can be derived by manipulating the unbiased estimator from
	# Song et al. (2007).
	return (
		xty - n / (n - 2.) * sum_squared_rows_x.dot(sum_squared_rows_y)
		+ squared_norm_x * squared_norm_y / ((n - 1) * (n - 2)))

def gram_linear(x):
  """Compute Gram (kernel) matrix for a linear kernel.

  Args:
    x: A num_examples x num_features matrix of features.

  Returns:
    A num_examples x num_examples Gram matrix of examples.
  """
  return x.dot(x.T)


def gram_rbf(x, threshold=1.0):
  """Compute Gram (kernel) matrix for an RBF kernel.

  Args:
    x: A num_examples x num_features matrix of features.
    threshold: Fraction of median Euclidean distance to use as RBF kernel
      bandwidth. (This is the heuristic we use in the paper. There are other
      possible ways to set the bandwidth; we didn't try them.)

  Returns:
    A num_examples x num_examples Gram matrix of examples.
  """
  dot_products = x.dot(x.T)
  sq_norms = np.diag(dot_products)
  sq_distances = -2 * dot_products + sq_norms[:, None] + sq_norms[None, :]
  sq_median_distance = np.median(sq_distances)
  return np.exp(-sq_distances / (2 * threshold ** 2 * sq_median_distance))


def center_gram(gram, unbiased=False):
  """Center a symmetric Gram matrix.

  This is equvialent to centering the (possibly infinite-dimensional) features
  induced by the kernel before computing the Gram matrix.

  Args:
    gram: A num_examples x num_examples symmetric matrix.
    unbiased: Whether to adjust the Gram matrix in order to compute an unbiased
      estimate of HSIC. Note that this estimator may be negative.

  Returns:
    A symmetric matrix with centered columns and rows.
  """
  if not np.allclose(gram, gram.T):
    raise ValueError('Input must be a symmetric matrix.')
  gram = gram.copy()

  if unbiased:
    # This formulation of the U-statistic, from Szekely, G. J., & Rizzo, M.
    # L. (2014). Partial distance correlation with methods for dissimilarities.
    # The Annals of Statistics, 42(6), 2382-2412, seems to be more numerically
    # stable than the alternative from Song et al. (2007).
    n = gram.shape[0]
    np.fill_diagonal(gram, 0)
    means = np.sum(gram, 0, dtype=np.float64) / (n - 2)
    means -= np.sum(means) / (2 * (n - 1))
    gram -= means[:, None]
    gram -= means[None, :]
    np.fill_diagonal(gram, 0)
  else:
    means = np.mean(gram, 0, dtype=np.float64)
    means -= np.mean(means) / 2
    gram -= means[:, None]
    gram -= means[None, :]

  return gram


def cka(gram_x, gram_y, debiased=False):
  """Compute CKA.

  Args:
    gram_x: A num_examples x num_examples Gram matrix.
    gram_y: A num_examples x num_examples Gram matrix.
    debiased: Use unbiased estimator of HSIC. CKA may still be biased.

  Returns:
    The value of CKA between X and Y.
  """
  gram_x = center_gram(gram_x, unbiased=debiased)
  gram_y = center_gram(gram_y, unbiased=debiased)

  # Note: To obtain HSIC, this should be divided by (n-1)**2 (biased variant) or
  # n*(n-3) (unbiased variant), but this cancels for CKA.
  scaled_hsic = gram_x.ravel().dot(gram_y.ravel())

  normalization_x = np.linalg.norm(gram_x)
  normalization_y = np.linalg.norm(gram_y)
  return scaled_hsic / (normalization_x * normalization_y)
