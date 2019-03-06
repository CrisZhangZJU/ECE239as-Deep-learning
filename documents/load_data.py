'''
Usage:
load single file:
train_loader, test_loader,val_loader = loader()(subject=1,
                                                batch_size= 20,
                                                num_validation = 37)
for all subjects: subject = "ALL"

Data should be placed in the same folder
'''

from torch.utils.data import TensorDataset
import torch
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit


class loader(object):

    def __init__(self):
        # self.path = None
        self.test_loader = None
        self.train_loader = None
        self.val_loader = None
        self.batch_size = None
        self.test_loaders = None
        self.X_train_valid_subs = None
        self.y_train_valid_subs = None
        self.X_test_subs = None
        self.y_test_subs = None
        self.num_validation = None

    def __call__(self, subject, batch_size=30, num_validation=37):

        # import data
        X_test = np.load("X_test.npy")
        y_test = np.load("y_test.npy")
        person_train_valid = np.load("person_train_valid.npy")
        X_train_valid = np.load("X_train_valid.npy")
        y_train_valid = np.load("y_train_valid.npy")
        person_test = np.load("person_test.npy")

        # get data from different subjects
        X_train_valid_subs = {}
        y_train_valid_subs = {}
        X_test_subs = {}
        y_test_subs = {}
        for i in range(person_train_valid.shape[0]):
            char_sub = "subject" + str(i + 1)
            X_train_valid_subs[char_sub] = X_train_valid[np.where(person_train_valid == i)[0], 0:22, :]
            y_train_valid_subs[char_sub] = y_train_valid[np.where(person_train_valid == i)[0]]
        for j in range(person_test.shape[0]):
            char_sub = "subject" + str(j + 1)
            X_test_subs[char_sub] = X_test[np.where(person_test == j)[0], 0:22, :]
            y_test_subs[char_sub] = y_test[np.where(person_test == j)[0]]

        self.batch_size = batch_size
        self.X_train_valid_subs = X_train_valid_subs
        self.y_train_valid_subs = y_train_valid_subs
        self.X_test_subs = X_test_subs
        self.y_test_subs = y_test_subs

        # make the loader
        self.getLoader(subject, num_validation)
        return self.train_loader, self.test_loader, self.val_loader


    def getLoader(self, subject, num_validation):


        if subject != 'ALL':
            subject_num = "subject" + str(subject)
            X_train_valid1 = self.X_train_valid_subs[subject_num]
            y_train_valid1 = self.y_train_valid_subs[subject_num]
            X_test1 = self.X_test_subs[subject_num]
            y_test1 = self.y_test_subs[subject_num]
        else:
            subject_num = "subject" + str(subject)
            X_train_valid1 = self.X_train_valid_subs['ALL']
            y_train_valid1 = self.y_train_valid_subs['ALL']
            X_test1 = self.X_test_subs['ALL']
            y_test1 = self.y_test_subs['ALL']

        y_train_valid1 -= np.amin(y_train_valid1)
        y_test1 -= np.amin(y_test1)

        # training set num
        num_training = y_train_valid1.shape[0] - num_validation


        sss = StratifiedShuffleSplit(n_splits=1, test_size=num_validation, random_state=0)
        for train_index, test_index in sss.split(X_train_valid1, y_train_valid1):
            X_train1, X_valid1 = X_train_valid1[train_index], X_train_valid1[test_index]
            y_train1, y_valid1 = y_train_valid1[train_index], y_train_valid1[test_index]


        print('Train data shape: ', X_train1.shape)
        print('Train labels shape: ', y_train1.shape)
        print('test data shape: ', X_test1.shape)
        print('test labels shape: ', y_test1.shape)
        print('Validation data shape: ', X_valid1.shape)
        print('Validation labels shape: ', y_valid1.shape)

        # Normailize the data set
        X_train1 = (X_train1 - np.mean(X_train1, axis=0)) / np.std(X_train1, axis=0)
        X_test1 = (X_test1 - np.mean(X_test1, axis=0)) / np.std(X_test1, axis=0)
        X_valid1 = (X_valid1 - np.mean(X_valid1, axis=0)) / np.std(X_valid1, axis=0)

        data_tensor = torch.Tensor(X_train1.reshape(num_training, 1, 22, 1000))
        target_tensor = torch.Tensor(y_train1)

        dataset = TensorDataset(data_tensor, target_tensor)

        # train
        self.train_loader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size,
                                                        shuffle=True, sampler=None,
                                                        batch_sampler=None,
                                                        num_workers=0,
                                                        pin_memory=False, drop_last=False,
                                                        timeout=0, worker_init_fn=None)

        data_tensor = torch.Tensor(X_test1.reshape(y_test1.shape[0], 1, 22, 1000))
        target_tensor = torch.Tensor(y_test1)

        # test
        dataset = TensorDataset(data_tensor, target_tensor)
        self.test_loader = torch.utils.data.DataLoader(dataset, batch_size=y_test1.shape[0],
                                                       shuffle=True, sampler=None,
                                                       batch_sampler=None,
                                                       num_workers=0,
                                                       pin_memory=False, drop_last=False,
                                                       timeout=0, worker_init_fn=None)

        # validation
        data_tensor = torch.Tensor(X_valid1.reshape(num_validation, 1, 22, 1000))
        target_tensor = torch.Tensor(y_valid1)

        dataset = TensorDataset(data_tensor, target_tensor)
        self.val_loader = torch.utils.data.DataLoader(dataset, batch_size=num_validation,
                                                      shuffle=True, sampler=None,
                                                      batch_sampler=None,
                                                      num_workers=0,
                                                      pin_memory=False, drop_last=False,
                                                      timeout=0, worker_init_fn=None)
