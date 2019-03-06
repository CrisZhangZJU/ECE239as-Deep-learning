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
        if subject != "ALL":
            self.getLoader(subject, num_validation)
        else:
            self.getAllDataSubject(num_validation)
        return self.train_loader, self.test_loader, self.val_loader, self.test_loaders

    def getAllDataSubject(self, num_validation):
        self.test_loaders = []

        X_train = np.array([])
        y_train = np.array([])
        X_test = np.array([])
        y_test = np.array([])
        X_val = np.array([])
        y_val = np.array([])
        for i in range(9):

            subject_num = "subject" + str(i+1)
            X_train_valid_temp = self.X_train_valid_subs[subject_num]
            y_train_valid_temp = self.y_train_valid_subs[subject_num]
            X_test_temp = self.X_test_subs[subject_num]
            y_test_temp = self.y_test_subs[subject_num]

            y_train_valid_temp -= np.amin(y_train_valid_temp)
            y_test_temp -= np.amin(y_test_temp)

            sss = StratifiedShuffleSplit(n_splits=1, test_size=num_validation, random_state=0)
            for train_index, test_index in sss.split(X_train_valid_temp, y_train_valid_temp):
                X_train_temp, X_val_temp = X_train_valid_temp[train_index], X_train_valid_temp[test_index]
                y_train_temp, y_val_temp = y_train_valid_temp[train_index], y_train_valid_temp[test_index]

            # stack all the X and y
            X_train = np.concatenate((X_train, X_train_temp), axis=0) if X_train.size else X_train_temp
            y_train = np.concatenate((y_train, y_train_temp)) if y_train.size else y_train_temp
            X_test = np.concatenate((X_test, X_test_temp), axis=0) if X_test.size else X_test_temp
            y_test = np.concatenate((y_test, y_test_temp)) if y_test.size else y_test_temp
            X_val = np.concatenate((X_val, X_val_temp), axis=0) if X_val.size else X_val_temp
            y_val = np.concatenate((y_val, y_val_temp)) if y_val.size else y_val_temp

        print('Train data shape: ', X_train.shape)
        print('Train labels shape: ', y_train.shape)
        print('test data shape: ', X_test.shape)
        print('test labels shape: ', y_test.shape)
        print('Validation data shape: ', X_val.shape)
        print('Validation labels shape: ', y_val.shape)
        # Normailize the data set
        X_train = (X_train - np.mean(X_train, axis=0)) / np.std(X_train, axis=0)
        X_test = (X_test - np.mean(X_test, axis=0)) / np.std(X_test, axis=0)
        X_val = (X_val - np.mean(X_val, axis=0)) / np.std(X_val, axis=0)

        data_tensor = torch.Tensor(X_train.reshape(y_train.shape[0], 1, 22, 1000))
        target_tensor = torch.Tensor(y_train)

        dataset = TensorDataset(data_tensor, target_tensor)

        # train
        self.train_loader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size,
                                                        shuffle=True, sampler=None,
                                                        batch_sampler=None,
                                                        num_workers=0,
                                                        pin_memory=False, drop_last=False,
                                                        timeout=0, worker_init_fn=None)

        data_tensor = torch.Tensor(X_test.reshape(y_test.shape[0], 1, 22, 1000))
        target_tensor = torch.Tensor(y_test)

        # test
        dataset = TensorDataset(data_tensor, target_tensor)
        self.test_loader = torch.utils.data.DataLoader(dataset, batch_size=50,
                                                       shuffle=True, sampler=None,
                                                       batch_sampler=None,
                                                       num_workers=0,
                                                       pin_memory=False, drop_last=False,
                                                       timeout=0, worker_init_fn=None)

        for i in range(9):
            subject_num1 = "subject" + str(i + 1)
            X_test1 = self.X_test_subs[subject_num1]
            y_test1 = self.y_test_subs[subject_num1]
            data_tensor = torch.Tensor(X_test1.reshape(y_test1.shape[0], 1, 22, 1000))
            target_tensor = torch.Tensor(y_test1)

            # test
            dataset = TensorDataset(data_tensor, target_tensor)
            self.test_loaders.append(torch.utils.data.DataLoader(dataset, batch_size=y_test1.shape[0],
                                                           shuffle=True, sampler=None,
                                                           batch_sampler=None,
                                                           num_workers=0,
                                                           pin_memory=False, drop_last=False,
                                                           timeout=0, worker_init_fn=None))

            # validation
        data_tensor = torch.Tensor(X_val.reshape(num_validation * 9, 1, 22, 1000))
        target_tensor = torch.Tensor(y_val)

        dataset = TensorDataset(data_tensor, target_tensor)
        self.val_loader = torch.utils.data.DataLoader(dataset, batch_size=num_validation,
                                                      shuffle=True, sampler=None,
                                                      batch_sampler=None,
                                                      num_workers=0,
                                                      pin_memory=False, drop_last=False,
                                                      timeout=0, worker_init_fn=None)

    def getLoader(self, subject, num_validation):


        if subject != 'ALL':
            subject_num = "subject" + str(subject)
            X_train_valid1 = self.X_train_valid_subs[subject_num]
            y_train_valid1 = self.y_train_valid_subs[subject_num]
            X_test1 = self.X_test_subs[subject_num]
            y_test1 = self.y_test_subs[subject_num]
        else:
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
