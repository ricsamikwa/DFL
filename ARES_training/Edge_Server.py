

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import threading
import tqdm
import time
import random
import numpy as np
from ARESopt.ARES_optimisation import BenchClient
import math
import logging
logging.basicConfig(level = logging.INFO,format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

import sys
sys.path.append('../')
from Wireless import *
import functions
import configurations

np.random.seed(0)
torch.manual_seed(0)

class Edge_Server(Wireless):
	def __init__(self, index, ip_address, server_port, model_name):
		super(Edge_Server, self).__init__(index, ip_address)
		self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
		self.port = server_port
		self.model_name = model_name
		self.sock.bind((self.ip, self.port))
		self.client_socks = {}

		while len(self.client_socks) < configurations.K:
			self.sock.listen(5)
			logger.info("CONNECTIONS.")
			(client_sock, (ip, port)) = self.sock.accept()
			logger.info('connection ' + str(ip))
			logger.info(client_sock)
			self.client_socks[str(ip)] = client_sock

		self.uninet = functions.get_model('Unit', self.model_name, configurations.model_len-1, self.device, configurations.model_cfg)
		self.uninet1 = functions.get_model('Unit', self.model_name, configurations.model_len-1, self.device, configurations.model_cfg)
		self.uninet2 = functions.get_model('Unit', self.model_name, configurations.model_len-1, self.device, configurations.model_cfg)
		self.w_local_list =[]
		self.groups = []
		#test dataset stuff

# 		self.transform_test = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
# ])
# 		self.testset = torchvision.datasets.CIFAR10(root=configurations.dataset_path, train=False, download=True, transform=self.transform_test)
# 		self.testloader = torch.utils.data.DataLoader(self.testset, batch_size=100, shuffle=False, num_workers=4)

		self.testloader = functions.get_test_dataloader_non_iid(4)

	def initialize(self, split_layers, offload,round, first, LR):
		if offload or first:
			self.split_layers = split_layers
			self.nets = {}
			self.optimizers= {}
			for i in range(len(split_layers)):
				client_ip = configurations.CLIENTS_LIST[i]
				if split_layers[i] < len(configurations.model_cfg[self.model_name]) -1: # Only offloading client need initialize optimizer in server
					self.nets[client_ip] = functions.get_model('Server', self.model_name, split_layers[i], self.device, configurations.model_cfg)

					#offloading weight in server also need to be initialized from the same global weight
					cweights = functions.get_model('Client', self.model_name, split_layers[i], self.device, configurations.model_cfg).state_dict()
					pweights = functions.split_weights_server(self.uninet.state_dict(),cweights,self.nets[client_ip].state_dict())
					self.nets[client_ip].load_state_dict(pweights)

					self.optimizers[client_ip] = optim.SGD(self.nets[client_ip].parameters(), lr=LR,
					  momentum=0.9)
				else:
					self.nets[client_ip] = functions.get_model('Server', self.model_name, split_layers[i], self.device, configurations.model_cfg)
			self.criterion = nn.CrossEntropyLoss()

		
			
		for i in self.client_socks:
			if round < 3:
				# msg = ['MSG_INITIAL_GLOBAL_WEIGHTS_SERVER_TO_CLIENT', self.w_local_list[i].state_dict()]
				msg = ['MSG_INITIAL_GLOBAL_WEIGHTS_SERVER_TO_CLIENT', self.uninet.state_dict()]
				self.send_msg(self.client_socks[i], msg)
			# else:

				# sending the correct one
				# we need to send only one!
				# check ip address to socket 
				# send the model that correspond to the group of the device
		if round >= 3:
			for p in range(len(self.groups)):
				for m in range(len(self.groups[p])):
					if p == 0:
						msg = ['MSG_INITIAL_GLOBAL_WEIGHTS_SERVER_TO_CLIENT', self.uninet1.state_dict()]
						self.send_msg(self.client_socks[self.groups[p][m]], msg)
					else:
						msg = ['MSG_INITIAL_GLOBAL_WEIGHTS_SERVER_TO_CLIENT', self.uninet2.state_dict()]
						self.send_msg(self.client_socks[self.groups[p][m]], msg)
			
			####################### current key bit
			########## how to determine the test set !!!
			############### mix everything. different test sets per group?

	def initialize_old(self, split_layers, offload,round, first, LR):
		if offload or first:
			self.split_layers = split_layers
			self.nets = {}
			self.optimizers= {}
			for i in range(len(split_layers)):
				client_ip = configurations.CLIENTS_LIST[i]
				if split_layers[i] < len(configurations.model_cfg[self.model_name]) -1: # Only offloading client need initialize optimizer in server
					self.nets[client_ip] = functions.get_model('Server', self.model_name, split_layers[i], self.device, configurations.model_cfg)

					#offloading weight in server also need to be initialized from the same global weight
					cweights = functions.get_model('Client', self.model_name, split_layers[i], self.device, configurations.model_cfg).state_dict()
					pweights = functions.split_weights_server(self.uninet.state_dict(),cweights,self.nets[client_ip].state_dict())
					self.nets[client_ip].load_state_dict(pweights)

					self.optimizers[client_ip] = optim.SGD(self.nets[client_ip].parameters(), lr=LR,
					  momentum=0.9)
				else:
					self.nets[client_ip] = functions.get_model('Server', self.model_name, split_layers[i], self.device, configurations.model_cfg)
			self.criterion = nn.CrossEntropyLoss()

		
			# here is where i continue 
			# send the right model 
			# keep ip address when receiving
			# send the right model 
		for i in self.client_socks:
			if round > 0 and round < 10:
				# msg = ['MSG_INITIAL_GLOBAL_WEIGHTS_SERVER_TO_CLIENT', self.w_local_list[i].state_dict()]
				msg = ['MSG_INITIAL_GLOBAL_WEIGHTS_SERVER_TO_CLIENT', self.uninet.state_dict()]
				self.send_msg(self.client_socks[i], msg)
			else:
				msg = ['MSG_INITIAL_GLOBAL_WEIGHTS_SERVER_TO_CLIENT', self.uninet.state_dict()]
				self.send_msg(self.client_socks[i], msg)


	def train(self, thread_number, client_ips):
		# Network test
		self.net_threads = {}
		for i in range(len(client_ips)):
			self.net_threads[client_ips[i]] = threading.Thread(target=self._thread_network_testing, args=(client_ips[i],))
			self.net_threads[client_ips[i]].start()

		for i in range(len(client_ips)):
			self.net_threads[client_ips[i]].join()

		self.bandwidth = {}
		for s in self.client_socks:
			msg = self.recv_msg(self.client_socks[s], 'MSG_TEST_NETWORK')
			self.bandwidth[msg[1]] = msg[2]

		# Training start
		self.threads = {}
		for i in range(len(client_ips)):
			if configurations.split_layer[i] == (configurations.model_len -1):
				self.threads[client_ips[i]] = threading.Thread(target=self._thread_training_no_offloading, args=(client_ips[i],))
				logger.info(str(client_ips[i]) + 'training start')
				self.threads[client_ips[i]].start()
			else:
				logger.info(str(client_ips[i]))
				self.threads[client_ips[i]] = threading.Thread(target=self._thread_training_offloading, args=(client_ips[i],))
				logger.info(str(client_ips[i]) + 'training start')
				self.threads[client_ips[i]].start()

		for i in range(len(client_ips)):
			self.threads[client_ips[i]].join()

		self.ttpi = {} # Training time per iteration
		for s in self.client_socks:
			msg = self.recv_msg(self.client_socks[s], 'MSG_TIME_ITERATION')
			self.ttpi[msg[1]] = msg[2]

		return self.bandwidth

	def _thread_network_testing(self, client_ip):
		msg = self.recv_msg(self.client_socks[client_ip], 'MSG_TEST_NETWORK')
		msg = ['MSG_TEST_NETWORK', self.uninet.cpu().state_dict()]
		self.send_msg(self.client_socks[client_ip], msg)

	def _thread_training_no_offloading(self, client_ip):
		pass

	def _thread_training_offloading(self, client_ip):
		#issues here!!
		# iteration = int((config.N / (config.K * config.B)))
		iteration = 50 # verify this number 50000/(5*100) = 100, but we have 50 iterations from the data ?
		# logger.info(str(iteration) + ' iterations!!')
		for i in range(iteration):
			msg = self.recv_msg(self.client_socks[client_ip], 'MSG_INTERMEDIATE_ACTIVATIONS_CLIENT_TO_SERVER')
			smashed_layers = msg[1]
			labels = msg[2]
			# logger.info(' received smashed data !!')
			inputs, targets = smashed_layers.to(self.device), labels.to(self.device)
			self.optimizers[client_ip].zero_grad()
			outputs = self.nets[client_ip](inputs)
			loss = self.criterion(outputs, targets)
			loss.backward()
			self.optimizers[client_ip].step()

			msg = ['MSG_INTERMEDIATE_GRADIENTS_SERVER_TO_CLIENT_'+str(client_ip), inputs.grad]
			self.send_msg(self.client_socks[client_ip], msg)

		logger.info(str(client_ip) + 'training end')
		return 'Done'
	
	def aggregate_old(self, client_ips,round):
		w_local_list =[]
		
		for i in range(len(client_ips)):
			msg = self.recv_msg(self.client_socks[client_ips[i]], 'MSG_SUB_WEIGHTS_CLIENT_TO_SERVER')
			if configurations.split_layer[i] != (configurations.model_len -1):
				w_local = (functions.concat_weights(self.uninet.state_dict(),msg[1],self.nets[client_ips[i]].state_dict()),configurations.N / configurations.K)
				w_local_list.append(w_local)
			else:
				w_local = (msg[1],configurations.N / configurations.K)
				w_local_list.append(w_local)
		zero_model = functions.zero_init(self.uninet).state_dict()
		
		
		aggregrated_model = functions.fed_avg(zero_model, w_local_list, configurations.N)
		
		self.uninet.load_state_dict(aggregrated_model)


		return aggregrated_model

	def aggregate(self, client_ips,round):
		w_local_list =[]
		# w_group1 = []
		# w_group2 = []
		groups = []

		# this mean I am receiving the client model in a aspecific order. 
		# i have to send model number i to device with address client_ips[i]
		# create a list of devices based on the perceived groups
		for i in range(len(client_ips)):
			msg = self.recv_msg(self.client_socks[client_ips[i]], 'MSG_SUB_WEIGHTS_CLIENT_TO_SERVER')
			if configurations.split_layer[i] != (configurations.model_len -1):
				w_local = (functions.concat_weights(self.uninet.state_dict(),msg[1],self.nets[client_ips[i]].state_dict()),configurations.N / configurations.K)
				w_local_list.append(w_local)
			else:
				w_local = (msg[1],configurations.N / configurations.K)
				w_local_list.append(w_local)
		zero_model = functions.zero_init(self.uninet).state_dict()
		
		# testing similarity before aggregrated_model
		# testing the model similarity for each layer
		# print(w_local_list[0])
		init_temp_one = w_local_list[0]
		init_temp_two = w_local_list[1]
		init_temp_three = w_local_list[2]
		# for phi in range(configurations.N):

		# if round == 0 or round ==10 or round == 99:	
		if round ==3:	
			cka_agreggate_12 = 0
			cka_agreggate_13 = 0
			for p in self.uninet.cpu().state_dict():
				# print(p)
				temp_one = init_temp_one[0][p]
				temp_two = init_temp_two[0][p]
				temp_three = init_temp_three[0][p]
				new_temp_one = temp_one.numpy().flatten()
				new_temp_two = temp_two.numpy().flatten()
				new_temp_three = temp_three.numpy().flatten()

				# print(new_temp_one)
				cka_from_features12 = functions.model_similarity_cka(new_temp_one, new_temp_two)
				cka_from_features13 = functions.model_similarity_cka(new_temp_one, new_temp_three)

				if not math.isnan(cka_from_features12):
					cka_agreggate_12 = cka_agreggate_12 + cka_from_features12
					cka_agreggate_13 = cka_agreggate_13 + cka_from_features13

				print('Linear CKA 12: {:.5f}'.format(cka_from_features12))
				print('Linear CKA 13: {:.5f}'.format(cka_from_features13))

			print('=> Agreggate CKA 12: {:.5f}'.format(cka_agreggate_12))
			print('=> Agreggate CKA 13: {:.5f}'.format(cka_agreggate_13))

		
			if cka_agreggate_12 > cka_agreggate_13:
				# group 1 has devices 1 and 2
				groups.append([client_ips[0],client_ips[1]])
				groups.append([client_ips[2]])
			else:
				# group2 has device 3
				groups.append([client_ips[0]])
				groups.append([client_ips[1],client_ips[2]])
				
			print("aggregrated 12 size : ", len(w_local_list[0:len(groups[0])]))
			print("aggregrated 12 size : ", len(w_local_list[len(groups[0]):len(groups[0])+len(groups[1])]) )

			self.groups = groups

			########################################################################
			# print("here we test again the agregated models")

			# temp_aggregrated_model1 = self.uninet1.state_dict()
			# temp_aggregrated_model2 = self.uninet2.state_dict()

			# for p in self.uninet1.cpu().state_dict():
			# 	# print(p)
			# 	################## start here !!!!!
			# 	###### what is the zero
			# 	temp_one = temp_aggregrated_model1[0][p]
			# 	temp_two = temp_aggregrated_model2[0][p]
				
			# 	new_temp_one = temp_one.numpy().flatten()
			# 	new_temp_two = temp_two.numpy().flatten()
				

			# 	# print(new_temp_one)
			# 	cka_from_features12 = functions.model_similarity_cka(new_temp_one, new_temp_two)
				
			# 	print('Linear CKA G1 and G2 : {:.5f}'.format(cka_from_features12))

			#########################################################################

		if round < 10:
			self.w_local_list = w_local_list

			return w_local_list

		

		
		aggregrated_model = functions.fed_avg(zero_model, w_local_list, configurations.N)
		aggregrated_model1 = functions.fed_avg(zero_model, w_local_list[0:len(self.groups[0])], configurations.N)
		aggregrated_model2 = functions.fed_avg(zero_model, w_local_list[len(self.groups[0]):len(self.groups[0])+len(self.groups[1])], configurations.N)
		
		self.uninet.load_state_dict(aggregrated_model)
		self.uninet1.load_state_dict(aggregrated_model1)
		self.uninet2.load_state_dict(aggregrated_model2)

		return aggregrated_model1

	def test(self, r):
		self.uninet.eval()
		test_loss = 0
		correct = 0
		total = 0
		with torch.no_grad():
			for batch_idx, (inputs, targets) in enumerate(tqdm.tqdm(self.testloader)):
				inputs, targets = inputs.to(self.device), targets.to(self.device)
				outputs = self.uninet(inputs)
				loss = self.criterion(outputs, targets)

				test_loss += loss.item()
				_, predicted = outputs.max(1)
				total += targets.size(0)
				correct += predicted.eq(targets).sum().item()

		acc = 100.*correct/total
		logger.info('Test Accuracy: {}'.format(acc))

		# Save checkpoint.
		torch.save(self.uninet.state_dict(), './'+ configurations.model_name +'.pth')

		return acc

	# The function to change more
	def adaptive_split(self, bandwidth):
		
		logger.info('Preparing Device')
		benchClient = BenchClient(1, '192.168.1.100', 50000, 'VGG', 6)

		offloading_strategy = benchClient.ARES_optimiser(0.6, bandwidth[configurations.CLIENTS_LIST[0]]) + 1
		print("Current Strategy: "+ str(offloading_strategy))
		# strategy configuration - refactoring
		configurations.split_layer = [1,3,4]
		logger.info('Next Round : ' + str(configurations.split_layer))

		msg = ['SPLIT_VECTOR',configurations.split_layer]
		self.scatter(msg)
		return configurations.split_layer


	def reinitialize(self, split_layers, offload,round, first, LR):
		self.initialize(split_layers, offload,round, first, LR)

	def scatter(self, msg):
		for i in self.client_socks:
			self.send_msg(self.client_socks[i], msg)
	
	