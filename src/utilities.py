"""
Classes for storing ML timing information.
"""
import torch
import numpy as np
import torch
import inspect
import os
import sys
import time

class Timer:
    """
    Internal class for recording timing information.
    """
    def __init__(self,
        name:   str,
        level:  str='epoch',
        type:   str='train',
        gpu:    bool=True,
    ):
        self.name = name
        self.level = level
        self.type = type
        self.gpu = gpu
        # initialized tensor
        self.timer_values = torch.empty(size=(0,1), dtype=torch.float)
        if self.gpu:
            self.timer_values.cuda()
            self.timer_start  = torch.cuda.Event(enable_timing=True)
            self.timer_end    = torch.cuda.Event(enable_timing=True)
            self.start = self._start_cuda
            self.end   = self._end_cuda
        else:
            self.timer_start = 0
            self.timer_end   = 0
            self.start = self._start_cpu
            self.end   = self._end_cpu
    
    def synchronize(self):
        torch.cuda.synchronize()

    def _start_cuda(self):
        self.timer_start.record()
    
    def _start_cpu(self):
        self.timer_start = time.time()
    
    def _end_cuda(self):
        self.timer_end.record()
        torch.cuda.synchronize()
        self.timer_values = torch.cat(
            (self.timer_values,
            torch.tensor([[self.timer_start.elapsed_time(self.timer_end)]])),
        )
    
    def _end_cpu(self):
        self.timer_end = time.time()

class Timers:
    """
    Collection of timers for ML tasks.
    """
    def __init__(self,
        gpu:    bool=True,
    ):
        self.gpu = gpu
        self.train_batch_params = {
            'type': 'training',
            'level':'batch',
            'gpu':  'self.gpu'
        }
        self.validation_batch_params = {
            'type': 'validation',
            'level':'batch',
            'gpu':  'self.gpu'
        }
        self.timers = {
            'epoch_training':   Timer('epoch_training', type='training', level='epoch',  gpu=self.gpu),
            'epoch_validation': Timer('epoch_validation', type='validation', level='epoch', gpu=self.gpu),
            # individual training information
            'training_data':            Timer('training_data',         **self.train_batch_params),
            'training_zero_grad':       Timer('training_zero_grad',    **self.train_batch_params),
            'training_forward':         Timer('training_forward',      **self.train_batch_params),
            'training_loss':            Timer('training_loss',         **self.train_batch_params),
            'training_loss_backward':   Timer('training_loss_backward',**self.train_batch_params),
            'training_backprop':        Timer('training_backprop',     **self.train_batch_params),
            'training_metrics':         Timer('training_metrics',      **self.train_batch_params),
            'training_progress':        Timer('training_progress',     **self.train_batch_params),
            'training_callbacks':       Timer('training_callbacks',    type='training', level='epoch',  gpu=self.gpu),
            # individual validation information
            'validation_data':      Timer('validation_data',      **self.validation_batch_params),
            'validation_forward':   Timer('validation_forward',   **self.validation_batch_params),
            'validation_loss':      Timer('validation_loss',      **self.validation_batch_params),
            'validation_metrics':   Timer('validation_metrics',   **self.validation_batch_params),
            'validation_progress':  Timer('validation_progress',  **self.validation_batch_params),
            'validation_callbacks': Timer('validation_callbacks', type='validation', level='epoch',  gpu=self.gpu),
        }

    def synchronize(self):
        torch.cuda.synchronize()

class MemoryTracker:
    """
    Internal class for recording memory information.
    """
    def __init__(self,
        name:   str,
        level:  str='epoch',
        type:   str='train',
        gpu:    bool=True,
    ):
        self.name = name
        self.level = level
        self.type = type
        self.gpu = gpu
        # initialized tensor
        self.memory_values = torch.empty(size=(0,1), dtype=torch.float)
        if self.gpu:
            self.memory_values.cuda()
            self.memory_start  = 0.0
            self.memory_end    = 0.0
            self.start = self._start_cuda
            self.end   = self._end_cuda
        else:
            self.memory_start = 0.0
            self.memory_end   = 0.0
            self.start = self._start_cpu
            self.end   = self._end_cpu
    
    def synchronize(self):
        torch.cuda.synchronize()

    def _start_cuda(self):
        self.memory_start = torch.cuda.memory_stats()['allocated_bytes.all.allocated']
    
    def _start_cpu(self):
        self.memory_start = 0.0
    
    def _end_cuda(self):
        self.memory_end = torch.cuda.memory_stats()['allocated_bytes.all.allocated']
        torch.cuda.synchronize()
        self.memory_values = torch.cat(
            (self.memory_values,
            torch.tensor([[self.memory_end - self.memory_start]])),
        )
    
    def _end_cpu(self):
        self.memory_end = 0.0

class MemoryTrackers:
    """
    Collection of memory_trackers for ML tasks.
    """
    def __init__(self,
        gpu:    bool=True,
    ):
        self.gpu = gpu
        self.train_batch_params = {
            'type': 'training',
            'level':'batch',
            'gpu':  'self.gpu'
        }
        self.validation_batch_params = {
            'type': 'validation',
            'level':'batch',
            'gpu':  'self.gpu'
        }
        self.memory_trackers = {
            'epoch_training':   MemoryTracker('epoch_training', type='training', level='epoch',  gpu=self.gpu),
            'epoch_validation': MemoryTracker('epoch_validation', type='validation', level='epoch', gpu=self.gpu),
            # individual training information
            'training_data':            MemoryTracker('training_data',         **self.train_batch_params),
            'training_zero_grad':       MemoryTracker('training_zero_grad',    **self.train_batch_params),
            'training_forward':         MemoryTracker('training_forward',      **self.train_batch_params),
            'training_loss':            MemoryTracker('training_loss',         **self.train_batch_params),
            'training_loss_backward':   MemoryTracker('training_loss_backward',**self.train_batch_params),
            'training_backprop':        MemoryTracker('training_backprop',     **self.train_batch_params),
            'training_metrics':         MemoryTracker('training_metrics',      **self.train_batch_params),
            'training_progress':        MemoryTracker('training_progress',     **self.train_batch_params),
            'training_callbacks':       MemoryTracker('training_callbacks',    type='training', level='epoch',  gpu=self.gpu),
            # individual validation information
            'validation_data':      MemoryTracker('validation_data',      **self.validation_batch_params),
            'validation_forward':   MemoryTracker('validation_forward',   **self.validation_batch_params),
            'validation_loss':      MemoryTracker('validation_loss',      **self.validation_batch_params),
            'validation_metrics':   MemoryTracker('validation_metrics',   **self.validation_batch_params),
            'validation_progress':  MemoryTracker('validation_progress',  **self.validation_batch_params),
            'validation_callbacks': MemoryTracker('validation_callbacks', type='validation', level='epoch',  gpu=self.gpu),
        }

    def synchronize(self):
        torch.cuda.synchronize()


"""
Various utility functions
"""

"""
Get the names of the arrays in an .npz file
"""
def get_array_names(
    input_file: str
):
    # check that file exists
    if not os.path.isfile(input_file):
        raise FileNotFoundError(f"Specified input file: '{input_file}' does not exist!")
    loaded_file = np.load(input_file)
    return list(loaded_file.files)

"""
The following function takes in a .npz file, and a set
of arrays specified by a dictionary, and appends them
to the .npz file, provided there are no collisions.
"""
def append_npz(
    input_file: str,
    arrays:     dict,
    override:   bool=False,
):
    # check that file exists
    if not os.path.isfile(input_file):
        raise FileNotFoundError(f"Specified input file: '{input_file}' does not exist!")
    if not isinstance(arrays, dict):
        raise ValueError(f"Specified array must be a dictionary, not '{type(arrays)}'!")
    # otherwise load file and check contents
    loaded_file = np.load(input_file, allow_pickle=True)
    loaded_arrays = {
        key: loaded_file[key] for key in loaded_file.files
    }
    # check that there are no identical array names if override set to false
    if override == False:
        for item in loaded_arrays.keys():
            if item in arrays.keys():
                raise ValueError(f"Array '{item}' already exists in .npz file '{input_file}'!")
    # otherwise add the array and save
    loaded_arrays.update(arrays)
    np.savez(
        input_file,
        **loaded_arrays
    )

"""
Get a list of arguments and default values for a method.
"""
def get_method_arguments(method):
    # argpase grabs input values for the method
    try:
        argparse = inspect.getfullargspec(method)
        args = argparse.args
        args.remove('self')
        default_params = [None for item in args]
        if argparse.defaults != None:
            for ii, value in enumerate(argparse.defaults):
                default_params[-(ii+1)] = value
        argdict = {item: default_params[ii] for ii, item in enumerate(args)}
        return argdict
    except:
        return {}

"""
Method for getting shapes of data and various other
useful information.
"""
def get_shape_dictionary(
    dataset=None,
    dataset_loader=None,
    model=None,
):
    data_shapes = {}
    # list of desired dataset values
    dataset_values = [
        'feature_shape',
        'class_shape',
    ]
    for item in dataset_values:
        try:
            data_shapes[item] = getattr(dataset, item)
        except:
            data_shapes[item] = 'missing'
    # list of desired dataloader values
    dataset_loader_values = [
        'num_total_train',
        'num_test',
        'num_train',
        'num_validation',
        'num_train_batches',
        'num_validation_batches',
        'num_test_batches',
    ]
    for item in dataset_loader_values:
        try:
            data_shapes[item] = getattr(dataset_loader, item)
        except:
            data_shapes[item] = 'missing'
    # list of desired model values
    model_values = [
        'input_shape',
        'output_shape',
    ]
    for item in model_values:
        try:
            data_shapes[item] = getattr(model, item)
        except:
            data_shapes[item] = 'missing'
    return data_shapes

def boxcar(
    x,
    mean:   float=0.11,
    sigma:  float=0.3,
    mode:   str='regular',
):  
    """
    Returns a value between -1 and ...
    depending on whether the values (x - (mean +- sigma))
    are < 0, == 0, or > 0.  If 
        a) regular :  return 0 if x < or > mean+-sigma
        b) regular :  return 1 if x > low but <= high
    """
    unit = torch.tensor([1.0])
    high = torch.heaviside(x - torch.tensor([mean + sigma]), unit)
    low = torch.heaviside(x - torch.tensor([mean - sigma]), unit)
    if mode == 'regular':
        return low - high
    else:
        return unit + high - low

def get_base_classes(derived):
    """
    Determine the base classes of some potentially inherited object.
    """
    bases = []
    try:
        for base in derived.__class__.__bases__:
            bases.append(base.__name__)
    except:
        pass
    return bases
