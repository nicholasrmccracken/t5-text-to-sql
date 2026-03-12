import os

import torch

import transformers
from transformers import T5ForConditionalGeneration, T5Config
from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS
import wandb

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def setup_wandb(args):
    # Implement this if you wish to use wandb in your experiments
    pass

def initialize_model(args):
    '''
    Initializes the T5 model either by loading pretrained weights for fine-tuning
    or by creating a new model from the same configuration with random initialization.

    Inputs:
        * args: Parsed command-line arguments containing model configuration settings.

    Returns:
        * model: T5 model moved to the appropriate device (CPU/GPU).
    '''
    if args.finetune:
        # Load pretrained T5 weights
        model = T5ForConditionalGeneration.from_pretrained('google-t5/t5-small')
    else:
        # Initialize a new T5 model with random weights
        config = T5Config.from_pretrained('google-t5/t5-small')
        model = T5ForConditionalGeneration(config)
        
    return model.to(DEVICE)

def mkdir(dirpath):
    if not os.path.exists(dirpath):
        try:
            os.makedirs(dirpath)
        except FileExistsError:
            pass

def save_model(checkpoint_dir, model, best):
    '''
    Saves the model parameters to a checkpoint file so the model can be reloaded later.

    Inputs:
        * checkpoint_dir (str): Directory where the checkpoint should be stored.
        * model: The T5 model whose parameters are being saved.
        * best (bool): Whether this checkpoint corresponds to the best-performing model.
    '''
    mkdir(checkpoint_dir)
    
    # Save the model parameters to disk
    model_path = os.path.join(checkpoint_dir, f"model_{best}.pt")
    torch.save(model.state_dict(), model_path)

def load_model_from_checkpoint(args, best):
    '''
    Loads a previously saved model checkpoint and restores the model parameters.

    Inputs:
        * args: Parsed command-line arguments used to determine the checkpoint location.
        * best (bool): Whether to load the best-performing checkpoint or the most recent one.

    Returns:
        * model: T5 model with parameters loaded from the checkpoint.
    '''
    model = initialize_model(args)
    
    # Determine the correct checkpoint directory
    model_type = "ft" if args.finetune else "scr"
    checkpoint_dir = os.path.join("checkpoints", f"{model_type}_experiments", args.experiment_name)
    
    # Load the saved weights
    model_path = os.path.join(checkpoint_dir, f"model_{best}.pt")
    state_dict = torch.load(model_path, map_location=DEVICE)
    
    # Restore weights into the model
    model.load_state_dict(state_dict)
    
    return model.to(DEVICE)

def initialize_optimizer_and_scheduler(args, model, epoch_length):
    optimizer = initialize_optimizer(args, model)
    scheduler = initialize_scheduler(args, optimizer, epoch_length)
    return optimizer, scheduler

def initialize_optimizer(args, model):
    decay_parameters = get_parameter_names(model, ALL_LAYERNORM_LAYERS)
    decay_parameters = [name for name in decay_parameters if "bias" not in name]
    optimizer_grouped_parameters = [
        {
            "params": [
                p for n, p in model.named_parameters() if (n in decay_parameters and p.requires_grad)
            ],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [
                p for n, p in model.named_parameters() if (n not in decay_parameters and p.requires_grad)
            ],
            "weight_decay": 0.0,
        },
    ]

    if args.optimizer_type == "AdamW":
        optimizer = torch.optim.AdamW(
            optimizer_grouped_parameters, lr=args.learning_rate, eps=1e-8, betas=(0.9, 0.999)
        )
    else:
        pass

    return optimizer
        
def initialize_scheduler(args, optimizer, epoch_length):
    num_training_steps = epoch_length * args.max_n_epochs
    num_warmup_steps = epoch_length * args.num_warmup_epochs

    if args.scheduler_type == "none":
        return None
    elif args.scheduler_type == "cosine":
        return transformers.get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps)
    elif args.scheduler_type == "linear":
        return transformers.get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps)
    else:
        raise NotImplementedError

def get_parameter_names(model, forbidden_layer_types):
    result = []
    for name, child in model.named_children():
        result += [
            f"{name}.{n}"
            for n in get_parameter_names(child, forbidden_layer_types)
            if not isinstance(child, tuple(forbidden_layer_types))
        ]
    # Add model specific parameters (defined with nn.Parameter) since they are not in any child.
    result += list(model._parameters.keys())
    return result
