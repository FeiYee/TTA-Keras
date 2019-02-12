import numpy as np

class TTA_ModelWrapper():
    def __init__(self, model, datagen):
        '''
        :param model: 模型
        :param datagen: 数据增强
        '''
        self.model = model
        self.datagen = datagen

    def predict(self, X, pre_num):
        '''
        :param X: Test数据
        :param pre_num: TTA数量
        :return:
        '''
#         self.datagen.fit(X)
        pred = []
        P = 0
        for i, tta in enumerate(self.datagen.flow(X, None, batch_size=1)):
            if i == 0:
                P += self.model.predict(X)
            else:
                P = np.add(P, self.model.predict(tta))
            if i == pre_num - 1:
                break
        P = P / pre_num
        return P
