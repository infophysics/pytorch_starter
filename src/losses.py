"""
Generic losses for tpc_ml.
"""
import torch
import torch.nn as nn

from logger import Logger

class GenericLoss:
    """
    """
    def __init__(self,
        name:   str='generic',
    ):
        self.name = name
        self.alpha = 0.0
        # set device to cpu for now
        self.device = 'cpu'

        # create empty tensors for batching
        self.batch_loss = torch.empty(size=(0,1), dtype=torch.float, device=self.device)

    def set_device(self,
        device
    ):  
        self.device = device
        self.reset_batch()

    def reset_batch(self):
        self.batch_loss = torch.empty(size=(0,1), dtype=torch.float, device=self.device)
    
    def loss(self,
        outputs,
        data,
    ):
        pass

"""
Container for generic losses
"""
class GenericLossList:
    """
    """
    def __init__(self,
        name:   str,
        losses:  list=[],
    ):
        self.name = name
        self.logger = Logger(self.name, file_mode="w")
        self.losses = losses

        # set to whatever the last call of set_device was.
        self.device = 'None'
    
    def set_device(self,
        device
    ):  
        for loss in self.losses:
            loss.set_device(device)
            loss.reset_batch()
        self.device = device

    def reset_batch(self):  
        for loss in self.losses:
            loss.reset_batch()

    def add_loss(self,
        loss:   GenericLoss
    ):
        self.losses.append(loss)
    
    def set_training_info(self,
        epochs: int,
        num_training_batches:   int,
        num_validation_batches:  int,
        num_test_batches:   int,
    ):
        for loss in self.losses:
            loss.set_training_info(
                epochs,
                num_training_batches,
                num_validation_batches,
                num_test_batches
            )
            loss.reset_batch()

    def loss(self,
        outputs,
        data,
    ):
        losses = [loss.loss(outputs, data) for loss in self.losses]
        return sum(losses)
    