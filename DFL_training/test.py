import torch
import torch.nn as nn
import torch.optim as optim
import tqdm
import time
import numpy as np
import sys

sys.path.append('../')
import configurations
import functions
from Wireless import *
import multiprocessing


import logging
logging.basicConfig(level = logging.INFO,format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

np.random.seed(0)
torch.manual_seed(0)

class Client(Wireless):
	def __init__(self, index, ip_address, datalen, model_name, split_layer):
		super(Client, self).__init__(index, ip_address)
		self.datalen = datalen
		self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
		self.model_name = model_name
		self.uninet = functions.get_model('Unit', self.model_name, configurations.model_len-1, self.device, configurations.model_cfg)

		# logger.info('Connecting to Server.')
		# self.sock.connect((server_addr,server_port))

	def initialize(self, split_layer, offload, first, LR):
		if offload or first:
			self.split_layer = split_layer

			logger.debug('Building Model.')
			self.net = functions.get_model('Client', self.model_name, self.split_layer, self.device, configurations.model_cfg)
			logger.debug(self.net)
			self.criterion = nn.CrossEntropyLoss()

		self.optimizer = optim.SGD(self.net.parameters(), lr=LR,
					  momentum=0.9)
		logger.debug('Receiving Global Weights..')

		#weights = self.recv_msg(self.sock)[1]
		# weights = functions.get_model('Unit', self.model_name, config.model_len-1, self.device, config.model_cfg)
		# if self.split_layer == (config.model_len -1):
		# 	self.net.load_state_dict(weights)
		# else:
		# 	pweights = functions.split_weights_client(weights,self.net.state_dict())
		# 	self.net.load_state_dict(pweights)
		# logger.debug('Initialize Finished')

	def train(self, trainloader):
		# Network speed test
		# network_time_start = time.time()
		# msg = ['MSG_TEST_NETWORK', self.uninet.cpu().state_dict()]
		# self.send_msg(self.sock, msg)
		# msg = self.recv_msg(self.sock,'MSG_TEST_NETWORK')[1]
		# network_time_end = time.time()
		# network_speed = (2 * config.model_size * 8) / (network_time_end - network_time_start) #Mbit/s 

		# logger.info('Network speed is {:}'.format(network_speed))
		# msg = ['MSG_TEST_NETWORK', self.ip, network_speed]
		# self.send_msg(self.sock, msg)

		# Training start
		s_time_total = time.time()
		time_training_c = 0
		self.net.to(self.device)

		
		# still testing the model
		initial_temp_one = self.net.cpu().state_dict()
		# new_temp_one = temp_one.numpy().flatten()


		self.net.train()
		if self.split_layer == (configurations.model_len -1): # No offloading training
			for batch_idx, (inputs, targets) in enumerate(tqdm.tqdm(trainloader)):
				inputs, targets = inputs.to(self.device), targets.to(self.device)
				self.optimizer.zero_grad()
				outputs = self.net(inputs)
				loss = self.criterion(outputs, targets)
				loss.backward()
				self.optimizer.step()
			
		else: # Offloading training
			for batch_idx, (inputs, targets) in enumerate(tqdm.tqdm(trainloader)):
				inputs, targets = inputs.to(self.device), targets.to(self.device)
				self.optimizer.zero_grad()
				outputs = self.net(inputs)

				# msg = ['MSG_LOCAL_ACTIVATIONS_CLIENT_TO_SERVER', outputs.cpu(), targets.cpu()]
				# self.send_msg(self.sock, msg)

				# # Wait receiving server gradients
				# gradients = self.recv_msg(self.sock)[1].to(self.device)

				outputs.backward(gradients)
				self.optimizer.step()

		e_time_total = time.time()
		logger.info('Total time: ' + str(e_time_total - s_time_total))

		training_time_pr = (e_time_total - s_time_total) / int((configurations.N / (configurations.K * configurations.B)))
		logger.info('training_time_per_iteration: ' + str(training_time_pr))

		# msg = ['MSG_TRAINING_TIME_PER_ITERATION', self.ip, training_time_pr]
		# self.send_msg(self.sock, msg)

		# testing the model similarity for each layer
		for p in self.net.cpu().state_dict():
			print(p)
			temp_one = self.net.cpu().state_dict()[p]
			temp_two = initial_temp_one[p]
			new_temp_one = temp_one.numpy().flatten()
			new_temp_two = temp_two.numpy().flatten()

			# print(new_temp_one)
			cka_from_features = functions.model_similarity_cka(new_temp_one, new_temp_two)
			print('Linear CKA from Features: {:.5f}'.format(cka_from_features))

		return e_time_total - s_time_total
		
	def upload(self):
		msg = ['MSG_LOCAL_WEIGHTS_CLIENT_TO_SERVER', self.net.cpu().state_dict()]
		self.send_msg(self.sock, msg)

	def reinitialize(self, split_layers, offload, first, LR):
		self.initialize(split_layers, offload, first, LR)

logger.info('Preparing Client')
client = Client(1, '192.168.5.22', 50002, 'VGG', 6)

offload = False
first = True # First initializaiton control
client.initialize(6, offload, first, configurations.LR)
first = False 

logger.info('Preparing Data.')
cpu_count = multiprocessing.cpu_count()
trainloader, classes= functions.get_local_dataloader(1, cpu_count)

# for r in range(configurations.R):
for r in range(1):

	logger.info('====================================>')
	logger.info('ROUND: {} START'.format(r))
	training_time = client.train(trainloader)
	logger.info('ROUND: {} END'.format(r))
	
	logger.info('==> Waiting for aggregration')
	#client.upload()

	logger.info('==> Reinitialization for Round : {:}'.format(r + 1))
	s_time_rebuild = time.time()
	if offload:
		configurations.split_layer = client.recv_msg(client.sock)[1]

	if r > 49:
		LR = configurations.LR * 0.1

	client.reinitialize(configurations.split_layer[1], offload, first, configurations.LR)
	e_time_rebuild = time.time()
	logger.info('Rebuild time: ' + str(e_time_rebuild - s_time_rebuild))
	logger.info('==> Reinitialization Finish')