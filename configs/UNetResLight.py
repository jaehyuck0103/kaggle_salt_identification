class CommonConfig:
    NET = 'UNetResLight'

    TRAIN_BATCH_SIZE = 16
    VALID_BATCH_SIZE = 32
    TEST_BATCH_SIZE = 32

    KFOLD_N = 5
    KFOLD_I_LIST = [0]


def TrainConfig2(A):
    A.NUM_EPOCH = 1000
    A.LEARNING_RATE = 0.0075

    A.SCHEDULER = 'Plateau'
    A.LR_DECAY_RATE = 0.5
    A.PATIENCE = 8
    A.EARLY_STOP = 16

    A.LOSS = 'LOVASZ'
    A.OPTIMIZER = 'SGD'

    return A


def TrainConfig3(A):
    A.NUM_EPOCH = 50
    A.LEARNING_RATE = 0.007

    A.SCHEDULER = 'Cosine'
    A.ETA_MIN = 0.0007
    A.EARLY_STOP = 1000000000

    A.LOSS = 'LOVASZ'
    A.OPTIMIZER = 'SGD'

    return A


class Config:
    COMMON = CommonConfig
    train_cycles = [TrainConfig2,
                    TrainConfig3, TrainConfig3, TrainConfig3,
                    TrainConfig3, TrainConfig3, TrainConfig3,
                    ]
    COMMON.CYCLE_N = len(train_cycles)

    def get_train_config(self, cycle):
        cfg = self.COMMON()
        cfg = self.train_cycles[cycle](cfg)
        return cfg

    def get_test_config(self):
        cfg = self.COMMON()
        return cfg
