import numpy as np
import keras

class DataGenerator(keras.utils.Sequence):
    def __init__(self, list_IDs, batch_size=32, n_pos1_classes=30, n_pos2_classes=32, n_super_classes=891, shuffle=True):
        self.batch_size = batch_size
        self.list_IDs = list_IDs
        self.n_pos1_classes = n_pos1_classes
        self.n_pos2_classes = n_pos2_classes
        self.n_super_classes = n_super_classes
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        return int(np.floor(len(self.list_IDs)/self.batch_size))

    def __getitem__(self, index):
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        list_IDs_tmp = [self.list_IDs[k] for k in indexes]
        
        emb, pos1, pos2, super = self.__data_generation(list_IDs_tmp)

        return emb, [keras.utils.to_categorical(pos1, num_classes=self.n_pos1_classes),\
                     keras.utils.to_categorical(pos2, num_classes=self.n_pos2_classes),\
                     keras.utils.to_categorical(pos2, num_classes=self.n_super_classes)]
#        return emb, {'pos1_output':keras.utils.to_categorical(pos1, num_classes=self.n_pos1_classes),\
#                     'pos2_output':keras.utils.to_categorical(pos2, num_classes=self.n_pos2_classes),
#                     'super_output':keras.utils.to_categorical(pos2, num_classes=self.n_super_classes)}
        
    def on_epoch_end(self):
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_tmp):
        em = []
        p1tmp = []
        p2tmp = []
        suptmp = []
        maxlen = 0
        for i, ID in enumerate(list_IDs_tmp):
            f = np.load('TLGbank/' + ID + '.npz')
            p1 = f['pos1']
            p2 = f['pos2']
            sup = f['super']
            e1 = f['embavg']
            l = len(p1)
            if l > maxlen:
                maxlen = l
            p1tmp.append(p1)
            p2tmp.append(p2)
            suptmp.append(sup)
            em.append(e1)
        
        p1 = np.zeros((self.batch_size, maxlen))
        for i in range(self.batch_size):
            for j in range(len(p1tmp[i])):
                p1[i][j] = p1tmp[i][j]

        p2 = np.zeros((self.batch_size, maxlen))
        for i in range(self.batch_size):
            for j in range(len(p2tmp[i])):
                p2[i][j] = p2tmp[i][j]

        super = np.zeros((self.batch_size, maxlen))
        for i in range(self.batch_size):
            for j in range(len(suptmp[i])):
                super[i][j] = suptmp[i][j]
                
        emb = np.zeros((self.batch_size, maxlen, 1024))
        for i in range(self.batch_size):
            for j in range(len(em[i])):
                emb[i][j] = em[i][0][j]
            
        return emb, p1, p2, super

