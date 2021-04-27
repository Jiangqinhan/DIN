import numpy as np


class DataInput:
    def __init__(self, data, batch_size):
        self.batch_size = batch_size
        self.data = data
        self.epoch_size = len(self.data) // batch_size
        if self.epoch_size * batch_size < len(self.data):
            self.epoch_size += 1
        self.i = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.i == self.epoch_size:
            raise StopIteration
        batch_data = self.data[self.i * self.batch_size:min(self.batch_size * (self.i+1), len(self.data))]
        self.i += 1
        userID, itemID, label, sequneceLength = [], [], [], []
        for sample in batch_data:
            userID.append(sample[0])
            itemID.append(sample[2])
            label.append(sample[3])
            sequneceLength.append(len(sample[1]))
        max_len = max(sequneceLength)
        # 不可以用batch_size 最后一个batch可能不满
        hist_item = np.zeros([len(batch_data), max_len], dtype=np.int64)

        k = 0
        for sample in batch_data:
            for l in range(len(sample[1])):
                hist_item[k][l] = sample[1][l]
            k += 1

        return self.i, (userID, itemID, label, hist_item, sequneceLength)
