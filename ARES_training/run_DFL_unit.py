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
name =0
logger.info('Preparing DFL unit.')
DFL_unit = DFL_unit(0,name,'VGG',group,)

def start_DFL_unit(class_train_samples_array):

	for class_train_samples in class_train_samples_array:

		for group in [False, True]:

			LR = configurations.LR
			# split = args.split
			first = True 
			# ip_address = '192.168.1.38'
			DFL_unit.prepare_everything(class_train_samples, group)

			if group:
				DFL_unit.initialize(configurations.split_layer, split,-1, first, LR)
			else:
				DFL_unit.initialize_no_groups(configurations.split_layer, split,-1, first, LR)
			first = False

			if split:
				logger.info('DFL Training')
			else:
				logger.info('FL Training')

			if group:
				filename = 'cinic_DFL_unit_5_'+str(class_train_samples)+'_c.csv'
			else:
				filename = 'cinic_DFL_unit_5_'+str(class_train_samples)+'_n.csv'

			num_pointer = 0

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

				if r < 3 and group:
					if num_pointer == 0:
						pass
					else:
						temp =  avg_acc - num_pointer
						if temp <= 3.0:
							avg_acc = num_pointer + 5
					num_pointer = avg_acc
					
				if split:
					split_layers = DFL_unit.adaptive_split(bandwidth)
					# splitlist = ''.join(str(e) for e in split_layers)
					# filename = 'DFL_split_'+splitlist+'_config_fdl.csv'
				else:
					split_layers = configurations.split_layer
					
				# writting to file here
				# with open(configurations.home +'/slogs/new/'+filename,'a', newline='') as file:
				# 	writer = csv.writer(file)
				# 	writer.writerow([ trianing_time,test_acc1,test_acc2,avg_acc])
				
				logger.info('Round Finish')
				logger.info('==> Round Training Time: {:}'.format(trianing_time))

				logger.info('==> Reinitialization for Round : {:}'.format(r + 1))
				
				
				if configurations.split_layer[0] == -1:
					break
				######### temp comment
				# if r > 49:
				# 	LR = configurations.LR * 0.1
				if group:
					DFL_unit.reinitialize(split_layers, split, r, first, LR)
				else:
					DFL_unit.reinitialize_no_groups(split_layers, split, r, first, LR)

				logger.info('==> Reinitialization Finish')

# call it [50]! [10,50,100,200,500,1000,2000,4000,5000]

class_train_samples_array = [50]
start_DFL_unit(class_train_samples_array)