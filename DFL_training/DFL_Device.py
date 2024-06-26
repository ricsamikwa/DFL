import torch
import socket
import time
import csv
import multiprocessing
import os
import argparse
import subprocess
import re

import logging
logging.basicConfig(level = logging.INFO,format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

import sys
sys.path.append('../')
from DFL_training.Device import Client
import configurations
import functions
# from threading import Thread

parser=argparse.ArgumentParser()
parser.add_argument('--split', help='ARES SPLIT', type= functions.str2bool, default= False)
args=parser.parse_args()


def get_any_ip_address():
    try:
        # Run the `ip` command to get all interface details
        result = subprocess.check_output(['ip', 'addr'], stderr=subprocess.STDOUT, universal_newlines=True)
        
        # Use regex to find all IPv4 addresses
        matches = re.findall(r'inet (\d+\.\d+\.\d+\.\d+)', result)
        if matches:
            for ip in matches:
                # Skip loopback address
                if not ip.startswith("127."):
                    return ip
            print("No non-loopback IP address found")
            return None
        else:
            print("No IP address found")
            return None
    except subprocess.CalledProcessError as e:
        print(f"Failed to get network interface details. Error: {e.output.strip()}")
        return None
    except Exception as e:
        print(f"Error: {e}")
        return None

# Get any IP address of the device
ip_address = get_any_ip_address()

hostname = socket.gethostname().replace('-desktop', '')
if ip_address== '192.168.1.51':
	hostname = 'nano12' # setting specific

ip_address = configurations.HOST2IP[hostname]
index = configurations.CLIENTS_CONFIG[ip_address]
datalen = configurations.N / configurations.K
split_layer = configurations.split_layer[index]
LR = configurations.LR

logger.info('Preparing Client')
client = Client(index, ip_address, configurations.SERVER_ADDR, configurations.SERVER_PORT, datalen, 'VGG', split_layer)

split = args.split
first = True # First initializaiton control
client.initialize(split_layer, split, -1, first, LR)
first = False 


logger.info('Preparing Data.')
cpu_count = multiprocessing.cpu_count()

trainloader, classes= functions.get_local_dataloader_non_iid(index, cpu_count)

if split:
	logger.info('DFL Training')
else:
	logger.info('Classic local Training')

flag = False # Bandwidth control flag.


def power_monitor_thread(stop):
	power = 0
	# power input
	filename =''+ hostname+'-'+str(configurations.split_layer[index])+'_power_config_2.csv'

	while True:

		if stop():
			break

		with open('/sys/bus/i2c/drivers/ina3221x/6-0040/iio:device0/in_power0_input') as t:
			power = ((t.read()))

		# print(power)	
		with open(configurations.home + '/slogs/' + filename,'a', newline='') as file:
			writer = csv.writer(file)
			writer.writerow([int(power)])
			
		time.sleep(0.5)
	
	
	return
	


def training_thread(LR):
	# print(hostname[0:3])
	# if hostname[0:4] == 'nano':
	# 	stop_threads = False
	# 	t1 = Thread(target=power_monitor_thread, args =(lambda : stop_threads,))
	# 	t1.start()

	for r in range(configurations.R):
		logger.info('====================================>')
		logger.info('ROUND: {} START'.format(r))

		#training time per round
		training_time,training_time_pr, network_speed, average_time = client.train(ip_address)

		if split:
			filename =''+ hostname+'-'+str(configurations.split_layer[index])+'_config_fdl.csv'
		else:
			filename = ''+ hostname+'_config_fdl.csv'


		logger.info('ROUND: {} END'.format(r))
		
		logger.info('==> Waiting for aggregration')
		client.upload()

		logger.info('==> Reinitialization for Round : {:}'.format(r + 1))
		s_time_rebuild = time.time()
		if split:
			configurations.split_layer = client.recv_msg(client.sock)[1]

		if r > 49:
			LR = configurations.LR * 0.1

		client.reinitialize(configurations.split_layer[index], split, r, first, LR)
		e_time_rebuild = time.time()
		logger.info('Rebuild time: ' + str(e_time_rebuild - s_time_rebuild))
		logger.info('==> Reinitialization Finish')
	
	# if hostname[0:3] == 'nano':
	# 	stop_threads = True
	# 	t1.join()
	# 	print('thread killed')
	

training_thread(LR)

# # create two new threads

# t2 = Thread(target=training_thread, args=(LR,))

# # start the threads
# t1.start()
# t2.start()

# # wait for the threads to complete
# t2.join()

# t1.join()
