"""
Class for a generic model trainer.
"""
import torch
import torch.nn as nn
import os
import sys
import numpy as np
from tqdm import tqdm
from logger import Logger
from loader import Loader
from metrics import GenericMetric
from utilities import Timers, MemoryTrackers
from callbacks import GenericCallbackList
from callbacks import GenericMetricCallback, GenericLossCallback
from callbacks import TimingCallback, MemoryTrackerCallback
import utilities as utils

"""
Allowed types one can use for a metric
"""
allowed_metric_types = [
    'train',
    'test',
    'both'
]

class GenericTrainer:
    """
    This class is an attempt to reduce code rewriting by putting together
    a set of functions that do everything that we could need with 
    respect to training.  There are a few objects which must be passed
    to the trainer, which include:
        (a) model     - an object which inherits from nn.Module
        (b) criterion - an object which has a defined function called "loss"
        (c) optimizer - some choice of optimizer, e.g. Adam
        (d) metrics   - (optional) an object which has certain defined functions
        (e) callbacks - (optional) an object which has certain defined functions 
    """
    def __init__(self,
        model,
        criterion,
        optimizer,
        metrics:        GenericMetric=None,
        callbacks:      GenericCallbackList=None,
        metric_type:    str='test',
        gpu:            bool=True,
        gpu_device:     int=0,
        seed:           int=0,
    ): 
        self.name = model.name + "_trainer"
        self.logger = Logger(self.name, output='both', file_mode='w')
        self.logger.info(f"constructing model trainer.")
        # Check for compatability with parameters

        # setup metrics
        if metric_type not in allowed_metric_types:
            self.logger.warning(f"metric type {metric_type} not in {allowed_metric_types}. Setting metric_type to 'test'.")
            metric_type = 'test'
        self.metric_type = metric_type

        # define directories
        self.predictions_dir = f'predictions/{model.name}/'
        self.manifold_dir    = f'plots/{model.name}/manifold/'
        self.features_dir    = f'plots/{model.name}/features/'
        self.timing_dir    = f'plots/{model.name}/timing/'
        self.memory_dir    = f'plots/{model.name}/memory/'

        # create directories
        if not os.path.isdir(self.predictions_dir):
            self.logger.info(f"creating predictions directory '{self.predictions_dir}'")
            os.makedirs(self.predictions_dir)
        if not os.path.isdir(self.manifold_dir):
            self.logger.info(f"creating manifold directory '{self.manifold_dir}'")
            os.makedirs(self.manifold_dir)
        if not os.path.isdir(self.features_dir):
            self.logger.info(f"creating features directory '{self.features_dir}'")
            os.makedirs(self.features_dir)
        if not os.path.isdir(self.timing_dir):
            self.logger.info(f"creating timing directory '{self.timing_dir}'")
            os.makedirs(self.timing_dir)
        if not os.path.isdir(self.memory_dir):
            self.logger.info(f"creating memory directory '{self.memory_dir}'")
            os.makedirs(self.memory_dir)

        # check for devices
        self.gpu = gpu
        self.gpu_device = gpu_device
        self.seed = seed
        
        if torch.cuda.is_available():
            self.logger.info(f"CUDA is available with devices:")
            for ii in range(torch.cuda.device_count()):
                device_properties = torch.cuda.get_device_properties(ii)
                cuda_stats = f"name: {device_properties.name}, "
                cuda_stats += f"compute: {device_properties.major}.{device_properties.minor}, "
                cuda_stats += f"memory: {device_properties.total_memory}"
                self.logger.info(f" -- device: {ii} - " + cuda_stats)

        # set gpu settings
        if self.gpu:
            if torch.cuda.is_available():
                if gpu_device >= torch.cuda.device_count() or gpu_device < 0:
                    self.logger.warn(f"desired gpu_device '{gpu_device}' not available, using device '0'")
                    self.gpu_device = 0
                self.device = torch.device(f"cuda:{self.gpu_device}")
                self.logger.info(f"CUDA is available, using device {self.gpu_device} - {torch.cuda.get_device_name(self.gpu_device)}")
            else:
                self.gpu == False
                self.logger.warn(f"CUDA not available! Using the cpu")
                self.device = torch.device("cpu")
        else:
            self.logger.info(f"using cpu as device")
            self.device = torch.device("cpu")
        
        # assign objects
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.metrics = metrics
        # TODO: Change this to a CallbackList, which is easier to maintain.
        if callbacks == None:
            # add generic callbacks
            self.callbacks = GenericCallbackList(
                name="default"
            )
        else:
            self.callbacks = callbacks

        # send other objects to the device
        self.model.set_device(self.device)

        # add callbacks for criterions
        self.criterion.set_device(self.device)
        self.criterion_callback = GenericLossCallback(
            self.criterion
        )
        self.callbacks.add_callback(self.criterion_callback)
        
        # add callbacks for metrics
        if self.metrics != None:
            self.metrics.set_device(self.device)
            self.metrics_callback = GenericMetricCallback(
                self.metrics,
            )
            self.callbacks.add_callback(self.metrics_callback)

        # add timing info
        self.timers = Timers(gpu=self.gpu)
        self.timer_callback = TimingCallback(
            self.timing_dir,
            self.timers
        )
        self.callbacks.add_callback(self.timer_callback)

        # add memory info
        self.memory_trackers = MemoryTrackers(gpu=self.gpu)
        self.memory_callback = MemoryTrackerCallback(
            self.memory_dir,
            self.memory_trackers
        )
        self.callbacks.add_callback(self.memory_callback)
    
    def __run_consistency_check(self,
        dataset_loader
    ):
        """
        This function performs various checks on the dataset, dataset_loader,
        model, criterion, metrics and callbacks to make sure that things are
        configured correctly and that the shapes of various tensors are also
        set correctly.

        Modify this accordingly
        """
        self.logger.info("passed consistency check.")

    def train(self,
        dataset_loader,             # dataset_loader to pass in
        epochs:     int=100,        # number of epochs to train
        checkpoint: int=10,         # epochs inbetween weight saving
        progress_bar:   str='all',  # progress bar from tqdm
        rewrite_bar:    bool=False, # wether to leave the bars after each epoch
        save_predictions:bool=True, # wether to save network outputs for all events to original file
        no_timing:  bool=False,     # wether to keep the bare minimum timing info as a callback
    ):
        """
        Main training loop.  First, we see if the user wants to omit timing information.
        """
        if (self.model.device != self.device):
            self.logger.error(f"device: '{self.device}' and model device: '{self.model.device}' are different!")
        if (self.criterion.device != self.device):
            self.logger.error(f"device: '{self.device}' and model device: '{self.criterion.device}' are different!")
        # TODO: run consistency check
        self.logger.info(f"running consistency check...")
        self.__run_consistency_check(dataset_loader)

        self.model.save_model(flag='init')
        # setting values in callbacks
        self.callbacks.set_device(self.device)
        self.callbacks.set_training_info(
            epochs,
            dataset_loader.num_train_batches,
            dataset_loader.num_validation_batches,
            dataset_loader.num_test_batches
        )
        # Training
        self.logger.info(f"training dataset '{dataset_loader.dataset.name}' for {epochs} epochs.")
        if no_timing:
            # TODO: Need to fix this so that memory and timing callbacks aren't called.
            self.__train_no_timing(
                dataset_loader,
                epochs,
                checkpoint,
                progress_bar,
                rewrite_bar,
                save_predictions
            )
        else:
            self.__train_with_timing(
                dataset_loader,
                epochs,
                checkpoint,
                progress_bar,
                rewrite_bar,
                save_predictions
            )

    def __train_with_timing(self,
        dataset_loader,             # dataset_loader to pass in
        epochs:     int=100,        # number of epochs to train
        checkpoint: int=10,         # epochs inbetween weight saving
        progress_bar:   str='all',  # progress bar from tqdm
        rewrite_bar:    bool=False, # wether to leave the bars after each epoch
        save_predictions:bool=True, # wether to save network outputs for all events to original file
    ):
        """
        Training usually consists of the following steps:
            (1) Zero-out training/validation/testing losses and metrics
            (2) Loop for N epochs:
                (a) Grab the current batch of (training/validation) data.
                (b) Run the data through the model and calculate losses/metrics.
                (c) Backpropagate the loss (training)
            (3) Evaluate the trained model on testing data.
        """
        # iterate over epochs
        for epoch in range(epochs):
            """
            Training stage.
            Setup the progress bar for the training loop.
            """
            if (progress_bar == 'all' or progress_bar == 'train'):
                training_loop = tqdm(
                    enumerate(dataset_loader.train_loader, 0), 
                    total=len(dataset_loader.train_loader), 
                    leave=rewrite_bar,
                    colour='green'
                )
            else:
                training_loop = enumerate(dataset_loader.train_loader, 0)

            # make sure to set model to train() during training!
            self.model.train()
            """            
            Setup timing/memory information for epoch.
            """
            self.timers.timers['epoch_training'].start()
            self.memory_trackers.memory_trackers['epoch_training'].start()
            self.timers.timers['training_data'].start()
            self.memory_trackers.memory_trackers['training_data'].start()
            for ii, data in training_loop:
                self.memory_trackers.memory_trackers['training_data'].end()
                self.timers.timers['training_data'].end()
                # zero the parameter gradients
                """
                There are choices here, either one can do:
                    model.zero_grad() or
                    optimizer.zero_grad() or
                    for param in model.parameters():        <== optimal choice
                        param.grad = None
                """
                self.timers.timers['training_zero_grad'].start()
                self.memory_trackers.memory_trackers['training_zero_grad'].start()
                for param in self.model.parameters():
                    param.grad = None
                self.memory_trackers.memory_trackers['training_zero_grad'].end()
                self.timers.timers['training_zero_grad'].end()
                # get the network output
                """
                The forward call takes in the entire data
                stream, which could have multiple inputs needed.
                It's up to the model to determine what to do with it.

                The forward call of the model could send out
                multiple output tensors, depending on the application
                (such as in an AE where the latent space values are
                important). It's up to the loss function to know what to expect.
                """
                self.timers.timers['training_forward'].start()
                self.memory_trackers.memory_trackers['training_forward'].start()
                outputs = self.model(data)
                self.memory_trackers.memory_trackers['training_forward'].end()
                self.timers.timers['training_forward'].end()

                # compute loss
                self.timers.timers['training_loss'].start()
                self.memory_trackers.memory_trackers['training_loss'].start()
                loss = self.criterion.loss(outputs, data)
                self.memory_trackers.memory_trackers['training_loss'].end()
                self.timers.timers['training_loss'].end()

                # backprop
                self.timers.timers['training_loss_backward'].start()
                self.memory_trackers.memory_trackers['training_loss_backward'].start()
                loss.backward()
                self.memory_trackers.memory_trackers['training_loss_backward'].end()
                self.timers.timers['training_loss_backward'].end()

                # record backprop timing
                self.timers.timers['training_backprop'].start()
                self.memory_trackers.memory_trackers['training_backprop'].start()
                self.optimizer.step()
                self.memory_trackers.memory_trackers['training_backprop'].end()
                self.timers.timers['training_backprop'].end()

                # update progress bar
                self.timers.timers['training_progress'].start()
                self.memory_trackers.memory_trackers['training_progress'].start()
                if (progress_bar == 'all' or progress_bar == 'train'):
                    training_loop.set_description(f"Training: Epoch [{epoch+1}/{epochs}]")
                    training_loop.set_postfix_str(f"loss={loss.item():.2e}")
                self.memory_trackers.memory_trackers['training_progress'].end()
                self.timers.timers['training_progress'].end()
                
                self.timers.timers['training_data'].start()
                self.memory_trackers.memory_trackers['training_data'].start()
            # update timing info
            self.memory_trackers.memory_trackers['epoch_training'].end()
            self.timers.timers['epoch_training'].end()
            self.model.eval()
            with torch.no_grad():
                """
                Run through a metric loop if there are any metrics
                defined.
                """
                if self.metrics != None:
                    if (progress_bar == 'all' or progress_bar == 'train'):
                        metrics_training_loop = tqdm(
                            enumerate(dataset_loader.train_loader, 0), 
                            total=len(dataset_loader.train_loader), 
                            leave=rewrite_bar,
                            colour='green'
                        )
                    else:
                        metrics_training_loop = enumerate(dataset_loader.train_loader, 0)
                    for ii, data in metrics_training_loop:
                        # update metrics
                        self.timers.timers['training_metrics'].start()
                        self.memory_trackers.memory_trackers['training_metrics'].start()
                        outputs = self.model(data)
                        self.metrics.update(outputs, data)
                        self.memory_trackers.memory_trackers['training_metrics'].end()
                        self.timers.timers['training_metrics'].end()
                        if (progress_bar == 'all' or progress_bar == 'train'):
                            metrics_training_loop.set_description(f"Training Metrics: Epoch [{epoch+1}/{epochs}]")
                
            # evaluate callbacks
            self.timers.timers['training_callbacks'].start()
            self.memory_trackers.memory_trackers['training_callbacks'].start()
            self.callbacks.evaluate_epoch(train_type='training')
            self.memory_trackers.memory_trackers['training_callbacks'].end()
            self.timers.timers['training_callbacks'].end()

            """
            Validation stage.
            Setup the progress bar for the validation loop.
            """
            if (progress_bar == 'all' or progress_bar == 'validation'):
                validation_loop = tqdm(
                    enumerate(dataset_loader.validation_loader, 0), 
                    total=len(dataset_loader.validation_loader), 
                    leave=rewrite_bar,
                    colour='blue'
                )
            else:
                validation_loop = enumerate(dataset_loader.validation_loader, 0)
            # make sure to set model to eval() during validation!
            self.model.eval()
            with torch.no_grad():
                """
                Setup timing information for epoch.
                """
                self.timers.timers['epoch_validation'].start()
                self.memory_trackers.memory_trackers['epoch_validation'].start()
                self.timers.timers['validation_data'].start()
                self.memory_trackers.memory_trackers['validation_data'].start()
                for ii, data in validation_loop:
                    self.memory_trackers.memory_trackers['validation_data'].end()
                    self.timers.timers['validation_data'].end()
                    # get the network output
                    self.timers.timers['validation_forward'].start()
                    self.memory_trackers.memory_trackers['validation_forward'].start()
                    outputs = self.model(data)
                    self.memory_trackers.memory_trackers['validation_forward'].end()
                    self.timers.timers['validation_forward'].end()

                    # compute loss
                    self.timers.timers['validation_loss'].start()
                    self.memory_trackers.memory_trackers['validation_loss'].start()
                    loss = self.criterion.loss(outputs, data)
                    self.memory_trackers.memory_trackers['validation_loss'].end()
                    self.timers.timers['validation_loss'].end()

                    # update progress bar
                    self.timers.timers['validation_progress'].start()
                    self.memory_trackers.memory_trackers['validation_progress'].start()
                    if (progress_bar == 'all' or progress_bar == 'validation'):
                        validation_loop.set_description(f"Validation: Epoch [{epoch+1}/{epochs}]")
                        validation_loop.set_postfix_str(f"loss={loss.item():.2e}")
                    self.memory_trackers.memory_trackers['validation_progress'].end()
                    self.timers.timers['validation_progress'].end()

                    self.timers.timers['validation_data'].start()
                    self.memory_trackers.memory_trackers['validation_data'].start()
                # update timing info
                self.memory_trackers.memory_trackers['epoch_validation'].end()
                self.timers.timers['epoch_validation'].end()
                """
                Run through a metric loop if there are any metrics
                defined.
                """
                if self.metrics != None:
                    if (progress_bar == 'all' or progress_bar == 'validation'):
                        metrics_validation_loop = tqdm(
                            enumerate(dataset_loader.validation_loader, 0), 
                            total=len(dataset_loader.validation_loader), 
                            leave=rewrite_bar,
                            colour='blue'
                        )
                    else:
                        metrics_validation_loop = enumerate(dataset_loader.validation_loader, 0)
                    for ii, data in metrics_validation_loop:
                        # update metrics
                        self.timers.timers['validation_metrics'].start()
                        self.memory_trackers.memory_trackers['validation_metrics'].start()
                        outputs = self.model(data)
                        self.metrics.update(outputs, data)
                        self.memory_trackers.memory_trackers['validation_metrics'].end()
                        self.timers.timers['validation_metrics'].end()
                        if (progress_bar == 'all' or progress_bar == 'validation'):
                            metrics_validation_loop.set_description(f"Validation Metrics: Epoch [{epoch+1}/{epochs}]")

            # evaluate callbacks
            self.timers.timers['validation_callbacks'].start()
            self.memory_trackers.memory_trackers['validation_callbacks'].start()
            self.callbacks.evaluate_epoch(train_type='validation')
            self.memory_trackers.memory_trackers['validation_callbacks'].end()
            self.timers.timers['validation_callbacks'].end()

            # save weights if at checkpoint step
            if epoch % checkpoint == 0:
                if not os.path.exists(".checkpoints/"):
                    os.makedirs(".checkpoints/")
                torch.save(
                    self.model.state_dict(), 
                    f".checkpoints/checkpoint_{epoch}.ckpt"
                )
            # free up gpu resources
            torch.cuda.empty_cache()
        # evaluate epoch callbacks
        self.callbacks.evaluate_training()
        self.logger.info(f"training finished.")
        """
        Testing stage.
        Setup the progress bar for the testing loop.
        We do not have timing information for the test
        loop stage, since it is generally quick
        and doesn't need to be optimized for any reason.
        """
        if (progress_bar == 'all' or progress_bar == 'test'):
            test_loop = tqdm(
                enumerate(dataset_loader.test_loader, 0), 
                total=len(dataset_loader.test_loader), 
                leave=rewrite_bar,
                colour='red'
            )
        else:
            test_loop = enumerate(dataset_loader.test_loader, 0)
        # make sure to set model to eval() during validation!
        self.model.eval()
        with torch.no_grad():
            for ii, data in test_loop:
                # get the network output
                outputs = self.model(data)

                # compute loss
                loss = self.criterion.loss(outputs, data)

                # update metrics
                if self.metrics != None:
                    self.metrics.update(outputs, data)

                # update progress bar
                if (progress_bar == 'all' or progress_bar == 'test'):
                    test_loop.set_description(f"Testing: Batch [{ii+1}/{dataset_loader.num_test_batches}]")
                    test_loop.set_postfix_str(f"loss={loss.item():.2e}")

            # evaluate callbacks
            self.callbacks.evaluate_epoch(train_type='test')
        self.callbacks.evaluate_testing()
        # save the final model
        self.model.save_model(flag='trained')

        # see if predictions should be saved
        if save_predictions:
            self.logger.info(f"Running inference to save predictions.")
            self.inference(
                dataset_loader,
                dataset_type='all',
                save_predictions=True,
            )
    
    def __train_no_timing(self,
        dataset_loader,             # dataset_loader to pass in
        epochs:     int=100,        # number of epochs to train
        checkpoint: int=10,         # epochs inbetween weight saving
        progress_bar:   str='all',  # progress bar from tqdm
        rewrite_bar:    bool=False, # wether to leave the bars after each epoch
        save_predictions:bool=True, # wether to save network outputs for all events to original file
    ):
        """
        No comments here since the code is identical to the __train_with_timing function 
        except for the lack of calls to timers.
        """
        for epoch in range(epochs):
            if (progress_bar == 'all' or progress_bar == 'train'):
                training_loop = tqdm(
                    enumerate(dataset_loader.train_loader, 0), 
                    total=len(dataset_loader.train_loader), 
                    leave=rewrite_bar,
                    colour='green'
                )
            else:
                training_loop = enumerate(dataset_loader.train_loader, 0)
            self.model.train()
            for ii, data in training_loop:
                for param in self.model.parameters():
                    param.grad = None
                outputs = self.model(data)
                loss = self.criterion.loss(outputs, data)
                loss.backward()
                self.optimizer.step()
                if (progress_bar == 'all' or progress_bar == 'train'):
                    training_loop.set_description(f"Training: Epoch [{epoch+1}/{epochs}]")
                    training_loop.set_postfix_str(f"loss={loss.item():.2e}")
            if self.metrics != None:
                if (progress_bar == 'all' or progress_bar == 'train'):
                    metrics_training_loop = tqdm(
                        enumerate(dataset_loader.train_loader, 0), 
                        total=len(dataset_loader.train_loader), 
                        leave=rewrite_bar,
                        colour='green'
                    )
                else:
                    metrics_training_loop = enumerate(dataset_loader.train_loader, 0)
                for ii, data in metrics_training_loop:
                    outputs = self.model(data)
                    self.metrics.update(outputs, data)
                    if (progress_bar == 'all' or progress_bar == 'train'):
                        metrics_training_loop.set_description(f"Training Metrics: Epoch [{epoch+1}/{epochs}]")
            self.callbacks.evaluate_epoch(train_type='training')
            if (progress_bar == 'all' or progress_bar == 'validation'):
                validation_loop = tqdm(
                    enumerate(dataset_loader.validation_loader, 0), 
                    total=len(dataset_loader.validation_loader), 
                    leave=rewrite_bar,
                    colour='blue'
                )
            else:
                validation_loop = enumerate(dataset_loader.validation_loader, 0)
            self.model.eval()
            with torch.no_grad():
                for ii, data in validation_loop:
                    outputs = self.model(data)
                    loss = self.criterion.loss(outputs, data)
                    if (progress_bar == 'all' or progress_bar == 'validation'):
                        validation_loop.set_description(f"Validation: Epoch [{epoch+1}/{epochs}]")
                        validation_loop.set_postfix_str(f"loss={loss.item():.2e}")
                if self.metrics != None:
                    if (progress_bar == 'all' or progress_bar == 'validation'):
                        metrics_validation_loop = tqdm(
                            enumerate(dataset_loader.validation_loader, 0), 
                            total=len(dataset_loader.validation_loader), 
                            leave=rewrite_bar,
                            colour='blue'
                        )
                    else:
                        metrics_validation_loop = enumerate(dataset_loader.validation_loader, 0)
                    for ii, data in metrics_validation_loop:
                        outputs = self.model(data)
                        self.metrics.update(outputs, data)
                        if (progress_bar == 'all' or progress_bar == 'validation'):
                            metrics_validation_loop.set_description(f"Validation Metrics: Epoch [{epoch+1}/{epochs}]")
            self.callbacks.evaluate_epoch(train_type='validation')
            if epoch % checkpoint == 0:
                if not os.path.exists(".checkpoints/"):
                    os.makedirs(".checkpoints/")
                torch.save(
                    self.model.state_dict(), 
                    f".checkpoints/checkpoint_{epoch}.ckpt"
                )
        self.callbacks.evaluate_training()
        self.logger.info(f"training finished.")
        if (progress_bar == 'all' or progress_bar == 'test'):
            test_loop = tqdm(
                enumerate(dataset_loader.test_loader, 0), 
                total=len(dataset_loader.test_loader), 
                leave=rewrite_bar,
                colour='red'
            )
        else:
            test_loop = enumerate(dataset_loader.test_loader, 0)
        self.model.eval()
        with torch.no_grad():
            for ii, data in test_loop:
                outputs = self.model(data)
                loss = self.criterion.loss(outputs, data)
                if self.metrics != None:
                    self.metrics.update(outputs, data)
                if (progress_bar == 'all' or progress_bar == 'test'):
                    test_loop.set_description(f"Testing: Batch [{ii+1}/{dataset_loader.num_test_batches}]")
                    test_loop.set_postfix_str(f"loss={loss.item():.2e}")
            self.callbacks.evaluate_epoch(train_type='test')
        self.callbacks.evaluate_testing()
        self.model.save_model(flag='trained')
        if save_predictions:
            self.logger.info(f"Running inference to save predictions.")
            self.inference(
                dataset_loader,
                dataset_type='all',
                save_predictions=True,
            )

    def inference(self,
        dataset_loader,             # dataset_loader to pass in
        dataset_type:   str='all',  # which dataset to use for inference
        save_predictions:bool=True, # wether to save the predictions
        progress_bar:   bool=True,  # progress bar from tqdm
        rewrite_bar:    bool=True, # wether to leave the bars after each epoch
    ):
        """
        Here we just do inference on a particular part
        of the dataset_loader, either 'train', 'validation',
        'test' or 'all'.
        """
        # check that everything is on the same device
        if (self.model.device != self.device):
            self.logger.error(f"device: '{self.device}' and model device: '{self.model.device}' are different!")
        if (self.criterion.device != self.device):
            self.logger.error(f"device: '{self.device}' and model device: '{self.criterion.device}' are different!")

        # determine loader
        if dataset_type == 'train':
            inference_loader = dataset_loader.train_loader
            num_batches = dataset_loader.num_training_batches
            inference_indices = dataset_loader.train_indices
        elif dataset_type == 'validation':
            inference_loader = dataset_loader.validation_loader
            num_batches = dataset_loader.num_validation_batches
            inference_indices = dataset_loader.validation_indices
        elif dataset_type == 'test':
            inference_loader = dataset_loader.test_loader
            num_batches = dataset_loader.num_test_batches
            inference_indices = dataset_loader.test_indices
        else:
            inference_loader = dataset_loader.all_loader
            num_batches = dataset_loader.num_all_batches
            inference_indices = dataset_loader.all_indices
        
        """
        Set up progress bar.
        """
        if (progress_bar == True):
            inference_loop = tqdm(
                enumerate(inference_loader, 0), 
                total=len(inference_loader), 
                leave=rewrite_bar,
                colour='magenta'
            )
        else:
            inference_loop = enumerate(inference_loader, 0)
        
        # set up array for predictions
        if self.num_output_elements == 1:
            predictions = torch.empty(size=(0,*self.output_shape), dtype=torch.float).to(self.device)
        else:
            predictions = [
                torch.empty(size=(0,*self.output_shape[ii]), dtype=torch.float).to(self.device)
                for ii in range(len(self.output_shape))
            ]
        
        self.logger.info(f"running inference on dataset '{dataset_loader.dataset.name}'.")
        # make sure to set model to eval() during validation!
        self.model.eval()
        with torch.no_grad():
            for ii, data in inference_loop:
                # get the network output
                outputs = self.model(data)

                # add predictions
                if self.num_output_elements == 1:
                    predictions = torch.cat((predictions, outputs),dim=0)
                else:
                    predictions = [
                        torch.cat((predictions[jj],outputs[jj]),dim=0)
                        for jj in range(len(self.output_shape))
                    ]

                # compute loss
                loss = self.criterion.loss(outputs, data)

                # update metrics
                if self.metrics != None:
                    self.metrics.update(outputs, data)

                # update progress bar
                if (progress_bar == True):
                    inference_loop.set_description(f"Inference: Batch [{ii+1}/{num_batches}]")
                    inference_loop.set_postfix_str(f"loss={loss.item():.2e}")
        if self.num_output_elements != 1:
            predictions = torch.stack(predictions)
        # save predictions if wanted
        if save_predictions:
            predictions_name = self.model.name + "_predictions"
            predictions_dict = {
                predictions_name: predictions.cpu(),
                predictions_name+'_indices': inference_indices
            }
            utils.append_npz(
                dataset_loader.dataset.input_file,
                predictions_dict,
                override=True,
            )
            self.logger.info(f"saved predictions array '{predictions_name}' to dataset file '{dataset_loader.dataset.input_file}'.")
        # evaluate callbacks
        self.callbacks.evaluate_inference()
        self.logger.info(f"returning predictions.")
        return predictions