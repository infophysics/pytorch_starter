"""
Example training code.
"""
import numpy as np
import os
from datetime import datetime

# project imports
from project.dataset import Dataset
from project.loss import Loss
from project.metric import Metric
from project.model import Model

from src.logger import Logger
from src.loader import Loader
from src.losses import GenericLossList
from src.metrics import GenericMetricList
from src.optimizer import Optimizer
from src.trainer import GenericTrainer

"""
Now we load our dataset as a torch dataset,
and then feed that into a dataloader.
"""
project_dataset = Dataset(
    name="project",
)
project_loader = Loader(
    project_dataset, 
    batch_size=256,
    test_split=0.3,
    test_seed=100,
    validation_split=0.3,
    validation_seed=100,
    num_workers=4
)

"""
Construct the project Model, specify the loss and the 
optimizer and metrics.
"""
project_config = {
    
}
project_model = Model(
    name = 'project_model',
    cfg  = project_config
) 

# create loss, optimizer and metrics
project_optimizer = Optimizer(
    model=project_model,
    optimizer='Adam'
)

# create criterions
project_loss = Loss(
    name="project_loss"
)
project_losses = GenericLossList(
    name="project_losses",
    losses=[project_loss]
)
# create metrics
project_metric = Metric(
    name="project_metric"
)
project_metrics = GenericMetricList(
    name="project_metrics",
    metrics=[project_metric]
)

# create trainer
project_trainer = GenericTrainer(
    model=project_model,
    criterion=project_losses,
    optimizer=project_optimizer,
    metrics=project_metrics,
    metric_type='test',
    gpu=True,
    gpu_device=0
)
project_trainer.train(
    project_loader,
    epochs=10,
)