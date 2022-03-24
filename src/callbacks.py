"""
Functions for evaluating and storing training information.
"""
import numpy as np
import torch
from matplotlib import pyplot as plt

from logger import Logger
from utilities import Timer, Timers
from utilities import MemoryTracker, MemoryTrackers
from losses import GenericLoss
from metrics import GenericMetric

class GenericCallback:
    """
    """
    def __init__(self):
        self.epochs = None
        self.num_training_batches = None
        self.num_validation_batches = None
        self.num_test_batches = None
        self.plot_colors = ['b','g','r','c','m','y']

        self.device = 'cpu'

    def set_device(self,
        device
    ):  
        self.device = device
    
    def reset_batch(self):
        pass

    def set_training_info(self,
        epochs: int,
        num_training_batches:   int,
        num_validation_batches:  int,
        num_test_batches:       int,
    ):
        self.epochs = epochs
        self.num_training_batches = num_training_batches
        self.num_validation_batches = num_validation_batches
        self.num_test_batches = num_test_batches
    
    def evaluate_epoch(self,
        train_type='training'
    ):
        pass

    def evaluate_training(self):
        pass

    def evaluate_testing(self):
        pass

    def evaluate_inference(self):
        pass

"""
Container for generic callbacks
"""
class GenericCallbackList:
    """
    """
    def __init__(self,
        name:   str,
        callbacks:  list=[],
    ):
        self.name = name
        self.logger = Logger(self.name, file_mode="w")
        self.callbacks = callbacks
    
    def set_device(self,
        device
    ):  
        for callback in self.callbacks:
            callback.set_device(device)
            callback.reset_batch()
        self.device = device

    def add_callback(self,
        callback:   GenericCallback
    ):
        self.callbacks.append(callback)
    
    def set_training_info(self,
        epochs: int,
        num_training_batches:   int,
        num_validation_batches:  int,
        num_test_batches:   int,
    ):
        for callback in self.callbacks:
            callback.set_training_info(
                epochs,
                num_training_batches,
                num_validation_batches,
                num_test_batches
            )

    def evaluate_epoch(self,
        train_type='train',
    ):
        if train_type not in ['training', 'validation', 'test']:
            self.logger.error(f"specified train_type: '{train_type}' not allowed!")
        for callback in self.callbacks:
            callback.evaluate_epoch(train_type)

    def evaluate_training(self):
        for callback in self.callbacks:
            callback.evaluate_training()

    def evaluate_testing(self):
        for callback in self.callbacks:
            callback.evaluate_testing()
    
    def evaluate_inference(self):
        for callback in self.callbacks:
            callback.evaluate_inference()

"""
Timing callback
"""
class TimingCallback(GenericCallback):
    """
    """
    def __init__(self,
        output_dir: str,
        timers: Timers
    ):
        super(TimingCallback, self).__init__()
        self.output_dir = output_dir
        self.timers = timers
    
    def evaluate_epoch(self,
        train_type='train'
    ):
        pass

    def evaluate_training(self):
        if self.epochs != None:
            if self.num_training_batches != None:
                self.__evaluate_training('training')
            if self.num_validation_batches != None:
                self.__evaluate_training('validation')
    
    def __evaluate_training(self, 
        train_type
    ):
        epoch_ticks = np.arange(1,self.epochs+1)
        if train_type == 'training':
            num_batches = self.num_training_batches
        else:
            num_batches = self.num_validation_batches

        batch_overhead = torch.tensor([0.0 for ii in range(self.epochs)])

        averages = {}
        stds = {}
        for item in self.timers.timers.keys():
            if len(self.timers.timers[item].timer_values) == 0:
                continue
            if self.timers.timers[item].type == train_type:
                if self.timers.timers[item].level == 'epoch':
                    temp_times = self.timers.timers[item].timer_values.squeeze()
                else:
                    temp_times = self.timers.timers[item].timer_values.reshape(
                        (self.epochs, num_batches)
                    ).sum(dim=1)
                averages[item] = temp_times.mean()
                stds[item] = temp_times.std()

        fig, axs = plt.subplots(figsize=(10,6))
        for item in self.timers.timers.keys():
            if len(self.timers.timers[item].timer_values) == 0:
                continue
            if self.timers.timers[item].type == train_type:
                if self.timers.timers[item].level == 'epoch':
                    temp_times = self.timers.timers[item].timer_values.squeeze()
                    linestyle='-'
                else:
                    temp_times = self.timers.timers[item].timer_values.reshape(
                        (self.epochs, num_batches)
                    ).sum(dim=1)
                    linestyle='--'
                axs.plot(
                    epoch_ticks, 
                    temp_times, 
                    linestyle=linestyle,  
                    label=f'{item.replace(f"{train_type}_","")}'
                )    
                axs.plot([], [],
                    marker='', linestyle='',
                    label=f"total: {temp_times.sum():.2f}ms"
                )
                if 'epoch' in item:
                    batch_overhead += temp_times
                elif 'callbacks' in item:
                    pass
                else:
                    batch_overhead -= temp_times
        axs.plot(epoch_ticks, batch_overhead, linestyle='-',  label='overhead')
        axs.plot([], [],
            marker='', linestyle='',
            label=f"total: {batch_overhead.sum():.2f}ms"
        )

        axs.set_xlabel("epoch")
        axs.set_ylabel(r"$\Delta t$ (ms)")
        axs.set_yscale('log')

        if train_type == 'training':
            plt.title(r"$\Delta t$ (ms) vs. epoch (training)")
        else:
            plt.title(r"$\Delta t$ (ms) vs. epoch (validation)")
        plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
        plt.tight_layout()
        if train_type == 'training':
            plt.savefig(f"{self.output_dir}/batch_training_timing.png")
        else:
            plt.savefig(f"{self.output_dir}/batch_validation_timing.png")
        
        fig, axs = plt.subplots(figsize=(10,6))
        box_values = torch.empty(size=(0,self.epochs))
        labels = []
        axs.plot([], [],
            marker='x', linestyle='',
            label=f'epochs: {self.epochs}'
        )
        for item in self.timers.timers.keys():
            if len(self.timers.timers[item].timer_values) == 0:
                continue
            if self.timers.timers[item].type == train_type:
                if self.timers.timers[item].level == 'epoch':
                    temp_times = self.timers.timers[item].timer_values.squeeze()
                    linestyle='-'
                else:
                    temp_times = self.timers.timers[item].timer_values.reshape(
                        (self.epochs, num_batches)
                    ).sum(dim=1)
                    linestyle='--'
                box_values = torch.cat((box_values, temp_times.unsqueeze(0)), dim=0)
                axs.plot([], [],
                    marker='', linestyle=linestyle,
                    label=f'{item.replace(f"{train_type}_","")}\n({averages[item]:.2f} +/- {stds[item]:.2f})'
                )
                labels.append(f'{item.replace(f"{train_type}_","")}')
        axs.boxplot(
            box_values,
            vert=True,
            patch_artist=True,
            labels=labels
        )    
        #axs.set_xlabel("epoch")
        axs.set_ylabel(r"$\langle\Delta t\rangle$ (ms)")
        axs.set_xticklabels(labels, rotation=45, ha='right')
        axs.set_yscale('log')

        if train_type == 'training':
            plt.title(r"$\langle\Delta t\rangle$ (ms) vs. task (training)")
        else:
            plt.title(r"$\langle\Delta t\rangle$ (ms) vs. task (validation)")
        plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
        plt.tight_layout()
        if train_type == 'training':
            plt.savefig(f"{self.output_dir}/batch_training_timing_avg.png")
        else:
            plt.savefig(f"{self.output_dir}/batch_validation_timing_avg.png")

    def evaluate_testing(self):
        pass
    
    def evaluate_inference(self):
        pass


"""
Timing callback
"""

class MemoryTrackerCallback(GenericCallback):
    """
    """
    def __init__(self,
        output_dir: str,
        memory_trackers: MemoryTrackers
    ):
        super(MemoryTrackerCallback, self).__init__()
        self.output_dir = output_dir
        self.memory_trackers = memory_trackers
    
    def evaluate_epoch(self,
        train_type='train'
    ):
        pass

    def evaluate_training(self):
        if self.epochs != None:
            if self.num_training_batches != None:
                self.__evaluate_training('training')
            if self.num_validation_batches != None:
                self.__evaluate_training('validation')
    
    def __evaluate_training(self, 
        train_type
    ):
        epoch_ticks = np.arange(1,self.epochs+1)
        if train_type == 'training':
            num_batches = self.num_training_batches
        else:
            num_batches = self.num_validation_batches

        batch_overhead = torch.tensor([0.0 for ii in range(self.epochs)])

        averages = {}
        stds = {}
        for item in self.memory_trackers.memory_trackers.keys():
            if len(self.memory_trackers.memory_trackers[item].memory_values) == 0:
                continue
            if self.memory_trackers.memory_trackers[item].type == train_type:
                if self.memory_trackers.memory_trackers[item].level == 'epoch':
                    temp_times = self.memory_trackers.memory_trackers[item].memory_values.squeeze()
                else:
                    temp_times = self.memory_trackers.memory_trackers[item].memory_values.reshape(
                        (self.epochs, num_batches)
                    ).sum(dim=1)
                averages[item] = temp_times.mean()
                stds[item] = temp_times.std()

        fig, axs = plt.subplots(figsize=(10,6))
        for item in self.memory_trackers.memory_trackers.keys():
            if len(self.memory_trackers.memory_trackers[item].memory_values) == 0:
                continue
            if self.memory_trackers.memory_trackers[item].type == train_type:
                if self.memory_trackers.memory_trackers[item].level == 'epoch':
                    temp_times = self.memory_trackers.memory_trackers[item].memory_values.squeeze()
                    linestyle='-'
                else:
                    temp_times = self.memory_trackers.memory_trackers[item].memory_values.reshape(
                        (self.epochs, num_batches)
                    ).sum(dim=1)
                    linestyle='--'
                axs.plot(
                    epoch_ticks, 
                    temp_times, 
                    linestyle=linestyle,  
                    label=f'{item.replace(f"{train_type}_","")}'
                )                  
                axs.plot([], [],
                    marker='', linestyle='',
                    label=f"total: {temp_times.sum():.2e}bytes"
                )
                if 'epoch' in item:
                    batch_overhead += temp_times
                elif 'callbacks' in item:
                    pass
                else:
                    batch_overhead -= temp_times
        axs.plot(epoch_ticks, batch_overhead, linestyle='-',  label='overhead')
        axs.plot([], [],
            marker='', linestyle='',
            label=f"total: {batch_overhead.sum():.2e}bytes"
        )

        axs.set_xlabel("epoch")
        axs.set_ylabel(r"$\Delta t$ (bytes)")
        axs.set_yscale('log')

        if train_type == 'training':
            plt.title(r"$\Delta m$ (bytes) vs. epoch (training)")
        else:
            plt.title(r"$\Delta m$ (bytes) vs. epoch (validation)")
        plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
        plt.tight_layout()
        if train_type == 'training':
            plt.savefig(f"{self.output_dir}/batch_training_memory.png")
        else:
            plt.savefig(f"{self.output_dir}/batch_validation_memory.png")

        fig, axs = plt.subplots(figsize=(10,6))
        box_values = torch.empty(size=(0,self.epochs))
        labels = []
        axs.plot([], [],
            marker='x', linestyle='',
            label=f'epochs: {self.epochs}'
        )
        for item in self.memory_trackers.memory_trackers.keys():
            if len(self.memory_trackers.memory_trackers[item].memory_values) == 0:
                continue
            if self.memory_trackers.memory_trackers[item].type == train_type:
                if self.memory_trackers.memory_trackers[item].level == 'epoch':
                    temp_times = self.memory_trackers.memory_trackers[item].memory_values.squeeze()
                    linestyle='-'
                else:
                    temp_times = self.memory_trackers.memory_trackers[item].memory_values.reshape(
                        (self.epochs, num_batches)
                    ).sum(dim=1)
                    linestyle='--'
                box_values = torch.cat((box_values, temp_times.unsqueeze(0)), dim=0)
                axs.plot([], [],
                    marker='', linestyle=linestyle,
                    label=f'{item.replace(f"{train_type}_","")}\n({averages[item]:.2f} +/- {stds[item]:.2f})'
                )
                labels.append(f'{item.replace(f"{train_type}_","")}')
        axs.boxplot(
            box_values,
            vert=True,
            patch_artist=True,
            labels=labels
        )    
        #axs.set_xlabel("epoch")
        axs.set_ylabel(r"$\langle\Delta m\rangle$ (bytes)")
        axs.set_xticklabels(labels, rotation=45, ha='right')
        axs.set_yscale('log')

        if train_type == 'training':
            plt.title(r"$\langle\Delta m\rangle$ (bytes) vs. task (training)")
        else:
            plt.title(r"$\langle\Delta m\rangle$ (bytes) vs. task (validation)")
        plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
        plt.tight_layout()
        if train_type == 'training':
            plt.savefig(f"{self.output_dir}/batch_training_memory_avg.png")
        else:
            plt.savefig(f"{self.output_dir}/batch_validation_memory_avg.png")

    def evaluate_testing(self):
        pass
    
    def evaluate_inference(self):
        pass

"""
Callback for recording loss information
"""
class GenericLossCallback(GenericCallback):
    """
    """
    def __init__(self,
        criterion_list,
    ):
        super(GenericLossCallback, self).__init__()

        self.criterion_list = criterion_list
        self.loss_names = [loss.name for loss in self.criterion_list.losses]

        # containers for training loss
        self.training_loss = torch.empty(
            size=(0,len(self.loss_names)), 
            dtype=torch.float, device=self.device
        )
        self.validation_loss = torch.empty(
            size=(0,len(self.loss_names)), 
            dtype=torch.float, device=self.device
        )
        self.test_loss = torch.empty(
            size=(0,len(self.loss_names)), 
            dtype=torch.float, device=self.device
        )
    
    def reset_batch(self):
        self.training_loss = torch.empty(
            size=(0,len(self.loss_names)), 
            dtype=torch.float, device=self.device
        )
        self.validation_loss = torch.empty(
            size=(0,len(self.loss_names)), 
            dtype=torch.float, device=self.device
        )
        self.test_loss = torch.empty(
            size=(0,len(self.loss_names)), 
            dtype=torch.float, device=self.device
        )

    def evaluate_epoch(self,
        train_type='training'
    ):  
        temp_losses = torch.empty(
            size=(1,0), 
            dtype=torch.float, device=self.device
        )
        # run through criteria
        if train_type == 'training':
            for ii in range(len(self.loss_names)):
                temp_loss = self.criterion_list.losses[ii].batch_loss.sum()/self.num_training_batches
                temp_losses = torch.cat(
                    (temp_losses,torch.tensor([[temp_loss]], device=self.device)),
                    dim=1
                )
            self.training_loss = torch.cat(
                (self.training_loss, temp_losses),
                dim=0
            )
        elif train_type == 'validation':
            for ii in range(len(self.loss_names)):
                temp_loss = self.criterion_list.losses[ii].batch_loss.sum()/self.num_validation_batches
                temp_losses = torch.cat(
                    (temp_losses,torch.tensor([[temp_loss]], device=self.device)),
                    dim=1
                )
            self.validation_loss = torch.cat(
                (self.validation_loss, temp_losses),
                dim=0
            )
        else:
            for ii in range(len(self.loss_names)):
                temp_loss = self.criterion_list.losses[ii].batch_loss.sum()/self.num_test_batches
                temp_losses = torch.cat(
                    (temp_losses,torch.tensor([[temp_loss]], device=self.device)),
                    dim=1
                )
            self.test_loss = torch.cat(
                (self.test_loss, temp_losses),
                dim=0
            )
        self.criterion_list.reset_batch()
    
    def evaluate_training(self):
        pass

    def evaluate_testing(self):
        epoch_ticks = np.arange(1, self.epochs+1)
        if self.num_training_batches != 0:
            fig, axs = plt.subplots(figsize=(10,5))
            if len(self.loss_names) > 1:
                final_training_value = f"(final={self.training_loss.sum(dim=1)[-1]:.2e})"
                axs.plot(
                    epoch_ticks,
                    self.training_loss.sum(dim=1).cpu().numpy(),
                    c='k',
                    label=rf"{'(total)':<12} {final_training_value:>16}"
                )
            for ii, loss in enumerate(self.loss_names):
                temp_loss = self.training_loss[:,ii]
                final_training_value = f"(final={temp_loss[-1]:.2e})"
                # plot using specified line colors
                axs.plot(
                    epoch_ticks,
                    temp_loss.cpu().numpy(),
                    c=self.plot_colors[ii],
                    label=rf"{loss:<12} {final_training_value:>16}"
                )
                axs.plot([],[],
                    marker='',
                    linestyle='',
                    label=rf"($\alpha=${self.criterion_list.losses[ii].alpha})"
                )
            axs.set_xlabel("epoch")
            axs.set_ylabel("loss")
            axs.set_yscale('log')
            plt.title("loss vs. epoch (training)")
            plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
            plt.tight_layout()
            plt.savefig("plots/epoch_training_loss.png")
        # validation plot
        if self.num_validation_batches != 0:
            fig, axs = plt.subplots(figsize=(10,5))
            if len(self.loss_names) > 1:
                final_validation_value = f"(final={self.validation_loss.sum(dim=1)[-1]:.2e})"
                axs.plot(
                    epoch_ticks,
                    self.validation_loss.sum(dim=1).cpu().numpy(),
                    c='k',
                    label=rf"{'(total)':<12} {final_validation_value:>16}"
                )
            for ii, loss in enumerate(self.loss_names):
                temp_loss = self.validation_loss[:,ii]
                final_validation_value = f"(final={temp_loss[-1]:.2e})"
                axs.plot(
                    epoch_ticks,
                    temp_loss.cpu().numpy(),
                    c=self.plot_colors[ii],
                    label=rf"{loss:<12} {final_validation_value:>16}"
                )
                axs.plot([],[],
                    marker='',
                    linestyle='',
                    label=rf"($\alpha=${self.criterion_list.losses[ii].alpha})"
                )
            axs.set_xlabel("epoch")
            axs.set_ylabel("loss")
            axs.set_yscale('log')
            plt.title("loss vs. epoch (validation)")
            plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
            plt.tight_layout()
            plt.savefig("plots/epoch_validation_loss.png")
        # plot both
        if self.num_training_batches != 0 and self.num_validation_batches != 0:
            fig, axs = plt.subplots(figsize=(10,5))
            final_value = f"(final={self.training_loss.sum(dim=1)[-1]:.2e})"
            axs.plot(
                epoch_ticks,
                self.training_loss.sum(dim=1).cpu().numpy(),
                c='k',
                linestyle='-',
                label=rf"{'(train)':<12} {final_value:>16}"
            )
            final_value = f"(final={self.validation_loss.sum(dim=1)[-1]:.2e})"
            axs.plot(
                epoch_ticks,
                self.validation_loss.sum(dim=1).cpu().numpy(),
                c='k',
                linestyle='--',
                label=rf"{'(validation)':<12} {final_value:>16}"
            )
            if len(self.loss_names) > 1:
                for ii, loss in enumerate(self.loss_names):
                    temp_training_loss = self.training_loss[:,ii]
                    temp_validation_loss = self.validation_loss[:,ii]
                    final_training_value = f"(final={temp_training_loss[-1]:.2e})"
                    final_validation_value = f"(final={temp_validation_loss[-1]:.2e})"
                    alpha_value = rf"($\alpha=${self.criterion_list.losses[ii].alpha})"
                    axs.plot(
                        epoch_ticks,
                        temp_training_loss.cpu().numpy(),
                        c=self.plot_colors[ii],
                        linestyle='-',
                        label=rf"{loss:<12} {final_training_value:>16}"
                    )
                    axs.plot(
                        epoch_ticks,
                        temp_validation_loss.cpu().numpy(),
                        c=self.plot_colors[ii],
                        linestyle='--',
                        label=rf"{alpha_value:<18} {final_validation_value:>16}"
                    )
            # plot test values
            if self.num_test_batches != 0:
                total_test_loss = f"{self.test_loss.sum(dim=1)[0]:.2e}"
                axs.plot([], [],
                    marker='x', 
                    c='k',
                    linestyle='',
                    label=f"{'(test) total:'} {total_test_loss}"
                )
                for ii, loss in enumerate(self.loss_names):
                    temp_loss_name = f"(test) {loss}:"
                    temp_loss_value = f"{self.test_loss[0][ii]:.2e}"
                    axs.plot([], [],
                        marker='x', 
                        c=self.plot_colors[ii],
                        linestyle='',
                        label=f"{temp_loss_name} {temp_loss_value}"
                    )
            axs.set_xlabel("epoch")
            axs.set_ylabel("loss")
            axs.set_yscale('log')
            plt.title("loss vs. epoch")
            plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
            plt.tight_layout()
            plt.savefig("plots/epoch_loss.png")
    
    def evaluate_inference(self):
        pass

"""
Generic metric callback
"""
class GenericMetricCallback(GenericCallback):
    """
    """
    def __init__(self,
        metrics_list
    ):  
        super(GenericMetricCallback, self).__init__()
        self.metrics_list = metrics_list
        if metrics_list != None:
            self.metric_names = [metric.name for metric in self.metrics_list.metrics]

        # containers for training metrics
        if metrics_list != None:
            # containers for training metric
            self.training_metrics = torch.empty(
                size=(0,len(self.metric_names)), 
                dtype=torch.float, device=self.device
            )
            self.validation_metrics = torch.empty(
                size=(0,len(self.metric_names)), 
                dtype=torch.float, device=self.device
            )
            self.test_metrics = torch.empty(
                size=(0,len(self.metric_names)), 
                dtype=torch.float, device=self.device
            )

    def reset_batch(self):
        self.training_metrics = torch.empty(
            size=(0,len(self.metric_names)), 
            dtype=torch.float, device=self.device
        )
        self.validation_metrics = torch.empty(
            size=(0,len(self.metric_names)), 
            dtype=torch.float, device=self.device
        )
        self.test_metrics = torch.empty(
            size=(0,len(self.metric_names)), 
            dtype=torch.float, device=self.device
        )

    def evaluate_epoch(self,
        train_type='training'
    ):  
        temp_metrics = torch.empty(
            size=(1,0), 
            dtype=torch.float, device=self.device
        )
        for ii in range(len(self.metric_names)):
            temp_metric = torch.tensor(
                [[self.metrics_list.metrics[ii].compute()]], 
                device=self.device
            )
            temp_metrics = torch.cat(
                (temp_metrics, temp_metric),
                dim=1
            )
        # run through criteria
        if train_type == 'training':
            self.training_metrics = torch.cat(
                (self.training_metrics, temp_metrics),
                dim=0
            )
        elif train_type == 'validation':
            self.validation_metrics = torch.cat(
                (self.validation_metrics, temp_metrics),
                dim=0
            )
        else:
            self.test_metrics = torch.cat(
                (self.test_metrics, temp_metrics),
                dim=0
            )
        self.metrics_list.reset_batch()

    def evaluate_training(self):
        pass

    def evaluate_testing(self):  
        # evaluate metrics from training and validation
        if self.metrics_list == None:
            return
        epoch_ticks = np.arange(1,self.epochs+1)
        # training plot
        fig, axs = plt.subplots(figsize=(10,5))
        if len(self.training_metrics) != 0:
            for ii, metric in enumerate(self.metric_names):
                temp_metric = self.training_metrics[:,ii]
                final_metric_value = f"(final={temp_metric[-1]:.2e})"
                axs.plot(
                    epoch_ticks,
                    temp_metric.cpu().numpy(),
                    c=self.plot_colors[-(ii+1)],
                    label=rf"{metric}"
                )
                axs.plot([],[],
                    marker='',
                    linestyle='',
                    label=rf"{final_metric_value}"
                )
            axs.set_xlabel("epoch")
            axs.set_ylabel("metric")
            axs.set_yscale('log')
            plt.title("metric vs. epoch (training)")
            plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
            plt.tight_layout()
            plt.savefig("plots/epoch_training_metrics.png")
        
        if len(self.validation_metrics) != 0:
            fig, axs = plt.subplots(figsize=(10,5))
            for ii, metric in enumerate(self.metric_names):
                temp_metric = self.validation_metrics[:,ii]
                final_metric_value = f"(final={temp_metric[-1]:.2e})"
                axs.plot(
                    epoch_ticks,
                    temp_metric.cpu().numpy(),
                    c=self.plot_colors[-(ii+1)],
                    label=rf"{metric}"
                )
                axs.plot([],[],
                    marker='',
                    linestyle='',
                    label=rf"{final_metric_value}"
                )
            axs.set_xlabel("epoch")
            axs.set_ylabel("metric")
            axs.set_yscale('log')
            plt.title("metric vs. epoch (validation)")
            plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
            plt.tight_layout()
            plt.savefig("plots/epoch_validation_metrics.png")

        if len(self.training_metrics) != 0 and len(self.validation_metrics) != 0:
            fig, axs = plt.subplots(figsize=(10,5))
            for ii, metric in enumerate(self.metric_names):
                temp_training_metric = self.training_metrics[:,ii]
                temp_validation_metric = self.validation_metrics[:,ii]
                final_training_metric_value = f"(final={temp_training_metric[-1]:.2e})"
                final_validation_metric_value = f"(final={temp_validation_metric[-1]:.2e})"
                axs.plot(
                    epoch_ticks,
                    temp_training_metric.cpu().numpy(),
                    c=self.plot_colors[-(ii+1)],
                    linestyle='-',
                    label=rf"{metric}"
                )
                axs.plot([],[],
                    marker='',
                    linestyle='',
                    label=rf"{final_training_metric_value}"
                )
                axs.plot(
                    epoch_ticks,
                    temp_validation_metric.cpu().numpy(),
                    c=self.plot_colors[-(ii+1)],
                    linestyle='--',
                    label=rf"{metric}"
                )
                axs.plot([],[],
                    marker='',
                    linestyle='',
                    label=rf"{final_validation_metric_value}"
                )
            if len(self.test_metrics) != 0:
                for ii, metric in enumerate(self.metric_names):
                    temp_metric = self.test_metrics[:,ii]
                    final_metric_value = f"(final={temp_metric[-1]:.2e})"
                    axs.plot([],[],
                        marker='x',
                        linestyle='',
                        c=self.plot_colors[-(ii+1)],
                        label=rf"(test) {metric}"
                    )
                    axs.plot([],[],
                    marker='',
                    linestyle='',
                    label=rf"{final_metric_value}"
                )
            axs.set_xlabel("epoch")
            axs.set_ylabel("metric")
            axs.set_yscale('log')
            plt.title("metric vs. epoch")
            plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
            plt.tight_layout()
            plt.savefig("plots/epoch_metrics.png")

    def evaluate_inference(self):
        pass