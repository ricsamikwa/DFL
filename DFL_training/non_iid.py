
# import sys
# sys.path.append('../')
# import functions
# import tqdm



# num_users_cifar = 10
# nclass_cifar = 2
# nsamples_cifar = 200
# rate_unbalance_cifar = 1.0

# train_dataset_cifar, test_dataset_cifar, user_groups_train_cifar, user_groups_test_cifar = functions.get_dataset_cifar10_extr_noniid(num_users_cifar, nclass_cifar, nsamples_cifar, rate_unbalance_cifar)

# # print(user_groups_train_cifar, user_groups_test_cifar)
# for batch_idx, (inputs, targets) in enumerate(tqdm.tqdm(user_groups_train_cifar)):
# 	 print(inputs, targets)