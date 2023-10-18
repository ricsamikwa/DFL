import time
import torch
import pickle
import csv
import argparse

import logging
logging.basicConfig(level = logging.INFO,format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

import sys
sys.path.append('../')
from ARES_training.DFL_unit import DFL_unit
import configurations
import functions

# parser=argparse.ArgumentParser()
# parser.add_argument('--split', help='ARES SPLIT', type= functions.str2bool, default= False)
# args=parser.parse_args()

# Create the ArgumentParser object
parser = argparse.ArgumentParser()

# Add the '--split' argument
parser.add_argument('--split', help='DFL SPLIT', type=functions.str2bool, default=False)

# Add the '--group' argument
parser.add_argument('--group', help='DFL Cluster', type=functions.str2bool, default=True)

# Parse the arguments
args = parser.parse_args()

# Get the values of the arguments
split = args.split
group = args.group

LR = configurations.LR
# split = args.split
first = True 
# ip_address = '192.168.1.38'
name =0
logger.info('Preparing DFL unit.')
DFL_unit = DFL_unit(0,name,'VGG',group)

if group:
	DFL_unit.initialize(configurations.split_layer, split,-1, first, LR)
else:
	DFL_unit.initialize_no_groups(configurations.split_layer, split,-1, first, LR)
first = False

#if split:
	#handle changes of split layers

if split:
	logger.info('DFL Training')
else:
	logger.info('FL Training')

# res = {}
# res['trianing_time'], res['test_acc_record'], res['bandwidth_record'] = [], [], []

for r in range(configurations.R):
	logger.info('====================================>')
	logger.info('==> Round {:} Start'.format(r))

	s_time = time.time()
	bandwidth = DFL_unit.train(thread_number= configurations.K, client_ips= configurations.CLIENTS_LIST)
	if group:
		DFL_unit.aggregate(configurations.CLIENTS_LIST,r)
	else:
		DFL_unit.aggregate_no_groups(configurations.CLIENTS_LIST,r)

	e_time = time.time()

	# Recording each round training time, bandwidth and test accuracy
	trianing_time = e_time - s_time
	# res['trianing_time'].append(trianing_time)
	# res['bandwidth_record'].append(bandwidth)

	if group:
		test_acc1, test_acc2 = DFL_unit.test(r)
		# res['test_acc_record'].append(test_acc)
	else:
		test_acc1, test_acc2 = DFL_unit.test_no_groups(r)
	avg_acc =  (test_acc1 + test_acc2)/2

	#temp item - WALK - senstive
	# config.split_layer[0] = config.split_layer[0] - 1
	
    #++++++++++++++++++++++++++++++++++++++

	if split:
		# ADAPT SPLIT LAYERS HERE!
		# split_layers = [2]
		# config.split_layer = split_layers
		split_layers = DFL_unit.adaptive_split(bandwidth)
		splitlist = ''.join(str(e) for e in split_layers)
		filename = 'DFL_split_'+splitlist+'_config_fdl.csv'
	else:
		split_layers = configurations.split_layer
		if group:
			filename = 'DFL_unit_c_10.csv'
		else:
			filename = 'DFL_unit_n_10.csv'

	# Here to start saving stuff
	with open(configurations.home +'/slogs/'+filename,'a', newline='') as file:
		writer = csv.writer(file)
		writer.writerow([ trianing_time,test_acc1,test_acc2,avg_acc])
    
	logger.info('Round Finish')
	logger.info('==> Round Training Time: {:}'.format(trianing_time))

	logger.info('==> Reinitialization for Round : {:}'.format(r + 1))
	
	
	if configurations.split_layer[0] == -1:
		break
	if r > 49:
		LR = configurations.LR * 0.1
	if group:
		DFL_unit.reinitialize(split_layers, split, r, first, LR)
	else:
		DFL_unit.reinitialize_no_groups(split_layers, split, r, first, LR)

	logger.info('==> Reinitialization Finish')
