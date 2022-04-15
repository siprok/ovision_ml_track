import pickle, os, time, gdown
import numpy as np

from tqdm import tqdm
from typing import List, Tuple
from config import Config as cfg
from .search import Base

import faiss

class SearchSolution(Base):

    def __init__(self, data_file='./data/train_data.pickle', 
                 data_url='https://drive.google.com/uc?id=1D_jPx7uIaCJiPb3pkxcrkbeFcEogdg2R') -> None:
        self.data_file = data_file
        self.data_url = data_url
        pass

    def set_base_from_pickle(self):
        if not os.path.isfile(self.data_file):
            if not os.path.isdir('./data'):
                os.mkdir('./data')
            gdown.download(self.data_url, self.data_file, quiet=False)

        with open(self.data_file, 'rb') as f:
            data = pickle.load(f)

        self.reg_matrix = [None] * len(data['reg'])
        self.ids = {}
        for i, key in enumerate(data['reg']):
            self.reg_matrix[i] = data['reg'][key][0][None]
            self.ids[i] = key

        self.reg_matrix = np.concatenate(self.reg_matrix, axis=0)
        self.pass_dict = data['pass']

        self.set_up_faiss()

    def set_up_faiss(self):
        dim = 512
        # print("... train index ...")
        self.index = faiss.index_factory(dim, 'IVF1000,Flat')
        self.index.train(self.reg_matrix.astype('float32'))
        # print("... add vectors to index ...")
        self.index.add(self.reg_matrix.astype('float32'))
        self.index.nprobe = 32

    def search(self, query: np.array) -> List[Tuple]:
        topn = 1
        return self.index.search(np.array([query]).astype('float32'), topn)[1] # Distance, Index

    def cos_sim(self, query: np.array) -> np.array:
        return np.dot(self.reg_matrix, query)

    def insert(self, feature: np.array) -> None:
        self.reg_matrix = np.concatenate(self.reg_matrix, feature, axis=0)
        # TODO: создать очередь элементов. Обновлять faiss только когда накопится критическая масса
        self.set_up_faiss()
