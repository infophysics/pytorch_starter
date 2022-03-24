"""
Generic metric class for tpc_ml.
"""
import torch
import torch.nn as nn

from logger import Logger

class GenericMetric:
    """
    """
    def __init__(self,
        name:   str='generic',
        output_shape:   tuple=(),
        target_shape:   tuple=(),
        input_shape:    tuple=(),
        when_compute:   str='all',
    ):
        self.name = name
        self.output_shape = output_shape
        self.target_shape = target_shape
        self.input_shape = input_shape
        self.when_compute = when_compute
        # set device to none for now
        self.device = 'cpu'

        # create empty tensors for evaluation
        self.batch_metric = torch.empty(
            size=(0,1), 
            dtype=torch.float, device=self.device
        )

    def reset_batch(self):
        self.batch_metric = torch.empty(
            size=(0,1), 
            dtype=torch.float, device=self.device
        )

    def update(self,
        outputs,
        data,
    ):
        pass

    def set_device(self,
        device
    ):  
        self.device = device
        self.reset_batch()

    def compute(self):
        pass

"""
Container for generic callbacks
"""
class GenericMetricList:
    """
    """
    def __init__(self,
        name:   str,
        output_shape:   tuple=(),
        target_shape:   tuple=(),
        metrics:  list=[],
    ):
        self.name = name
        self.output_shape = output_shape
        self.target_shape = target_shape
        self.logger = Logger(self.name, file_mode="w")
        self.metrics = metrics

    def set_device(self,
        device
    ):  
        for metric in self.metrics:
            metric.set_device(device)
            metric.reset_batch()
        self.device = device
    
    def set_shapes(self,
        output_shape,
        target_shape,
        input_shape=(),
    ):
        for metric in self.metrics:
            metric.output_shape = output_shape
            metric.target_shape = target_shape
            metric.input_shape = input_shape
            metric.reset_batch()

    def reset_batch(self):  
        for metric in self.metrics:
            metric.reset_batch()

    def add_metric(self,
        metric:   GenericMetric
    ):
        self.metrics.append(metric)
    
    def set_training_info(self,
        epochs: int,
        num_training_batches:   int,
        num_validation_batches:  int,
        num_test_batches:   int,
    ):
        for metric in self.metrics:
            metric.set_training_info(
                epochs,
                num_training_batches,
                num_validation_batches,
                num_test_batches
            )
    
    def update(self,
        outputs,
        data,
        train_type: str='all',
    ):
        for metric in self.metrics:
            if train_type == metric.when_compute or metric.when_compute == 'all':
                metric.update(outputs, data)
    
    def compute(self,
        outputs,
        data
    ):
        metrics = [metric.compute(outputs, data) for metric in self.metrics]
        return metrics
