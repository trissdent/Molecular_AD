import torch

class OptimizerHandler:
    """
    Optimizer and scheduler handler.
    Modify this for your task.
    """
    def __init__(self, optimizer_type="adam", lr=1e-3, weight_decay=0, scheduler_type=None, **scheduler_kwargs):
        self.optimizer_type = optimizer_type
        self.lr = lr
        self.weight_decay = weight_decay
        self.scheduler_type = scheduler_type
        self.scheduler_kwargs = scheduler_kwargs
    
    def get_optimizer(self, model_parameters):
        if self.optimizer_type == "adam":
            return torch.optim.Adam(model_parameters, lr=self.lr)
        elif self.optimizer_type == "adamw":
            return torch.optim.AdamW(model_parameters, lr=self.lr, weight_decay=self.weight_decay)
        elif self.optimizer_type == "sgd":
            return torch.optim.SGD(model_parameters, lr=self.lr, momentum=0.9, weight_decay=self.weight_decay)
        else:
            raise ValueError(f"Unknown optimizer: {self.optimizer_type}")
    
    def get_scheduler(self, optimizer):
        if self.scheduler_type is None:
            return None
        elif self.scheduler_type == "step":
            return torch.optim.lr_scheduler.StepLR(
                optimizer,
                step_size=self.scheduler_kwargs.get("step_size", 10),
                gamma=self.scheduler_kwargs.get("gamma", 0.1)
            )
        elif self.scheduler_type == "cosine":
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=self.scheduler_kwargs.get("T_max", 100)
            )
        elif self.scheduler_type == "plateau":
            return torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode="min",
                patience=self.scheduler_kwargs.get("patience", 5)
            )
        else:
            raise ValueError(f"Unknown scheduler: {self.scheduler_type}")