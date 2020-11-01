# File: train_maml
# ----------------
# Training script for models using MAML.

import argparse
import torch
import torch.nn.functional as F
import numpy as np
import logging
import time

import utils
import constants
from dataset.task_dataset import TaskHandler
from models.model_utils import initialize_model, save_model

def get_arguments():
    '''
    Uses argparse to get the arguments for the experiment
    '''
    
    parser = argparse.ArgumentParser(description="Few-shot music generation with MAML")

    # Optimization arguments
    parser.add_argument("--num_train_iterations", type=int, default=constants.NUM_TRAIN_ITERATIONS,
                        help="Number of meta-training steps we will take in training")
    parser.add_argument("--num_inner_updates", type=int, default=constants.NUM_INNER_UPDATES,
                        help="Number of inner updates MAML will take in the inner loop")
    parser.add_argument("--outer_lr", type=float, default=constants.OUTER_LR,
                        help="The learning rate used for the outer optimization")
    parser.add_argument("--inner_lr", type=float, default=constants.INNER_LR,
                        help="The learning rate used for the inner optimization")

    # Model architecture arguments
    parser.add_argument("--embed_dim", type=int, default=constants.EMBED_DIM,
                        help="Embedding dimension for simple LSTM")
    parser.add_argument("--hidden_dim", type=int, default=constants.HIDDEN_DIM,
                        help="Hidden dimension for simple LSTM")

    # Data loading arguments
    parser.add_argument("--num_support", type=int, default=constants.NUM_SUPPORT,
                        help="Number of support snippets given to the model")
    parser.add_argument("--num_query", type=int, default=constants.NUM_QUERY,
                        help="Number of query snippets given to the model")
    parser.add_argument("--meta_batch_size", type=int, default=constants.META_BATCH_SIZE,
                        help="Number of tasks sampled at each meta-training step")
    parser.add_argument("--num_workers", type=int, default=constants.NUM_WORKERS,
                        help="Number of threads to use in the data loader")
    parser.add_argument("--context_len", type=int, default=constants.CONTEXT_LEN,
                        help="The length of the training snippets")
    parser.add_argument("--test_prefix_len", type=int, default=constants.TEST_PREFIX_LEN,
                        help="The length of the test snippets")
    
    # Miscellaneous evaluation and checkpointing arguments
    parser.add_argument("--model_type", type=str, default="SimpleLSTM", choices=constants.MODEL_TYPES,
                        help="The name of the model class to be used")
    parser.add_argument("--report_train_every", type=int, default=constants.REPORT_TRAIN_EVERY,
                        help="Report the training accuracy every report_train_every iterations")
    parser.add_argument("--evaluate_every", type=int, default=constants.EVALUATE_EVERY,
                        help="Compute validation accuracy every evaluate_every iterations")
    parser.add_argument("--save_checkpoint_every", type=int, default=constants.SAVE_CHECKPOINT_EVERY,
                        help="Save a model checkpoint every save_checkpoint_every iterations")
    parser.add_argument("--load_from_iteration", type=int, default=-1,
                        help="Initialize the model with a checkpoint from the provided iteration."\
                        +"Setting -1 will start the model from scratch")

    # Experiment arguments
    parser.add_argument("--experiment_name", type=str, default="test",
                        help="The name of the experiment (folder). This is where checkpoints, plots"\
                        +"and logs will reside.")
    parser.add_argument("--log_name", type=str, default="train",
                        help="The name of the logging file")
    parser.add_argument("--seed", type=int, default=-1,
                        help="Random seed for the experiment. -1 indicates that no seed be set")    

    # Parse and return
    args = parser.parse_args()
    return args

def train(model, dataloader, device, args):
    '''
    Outer training loop for MAML
    '''
    # Initialize the optimizer
    outer_optimizer = torch.optim.Adam(model.parameters(), lr=args.outer_lr)

    # Initialize the validation accuracy list
    validation_accs = []

    # Perform the outer updates
    for iteration in range(args.num_train_iterations):
        # Train step
        accuracy = outer_maml_step(model, outer_optimizer, dataloader, device, args, "train")

        # Report training accuracy
        if (iteration + 1) % args.report_train_every == 0:
            logging.info("Train Accuracy for Iteration {}/{}: {}".format(iteration + 1, args.num_train_iterations,
                                                                         accuracy))

        # Perform validation
        if (iteration + 1) % args.evaluate_every == 0:
            accuracy = outer_maml_step(model, outer_optimizer, dataloader, device, args, "val")
            logging.info("Validation Accuracy for Iteration {}/{}: {}".format(iteration + 1, args.num_train_iterations,
                                                                              accuracy))
            validation_accs.append(accuracy)

        # Save the model
        if (iteration + 1) & args.save_checkpoint_every == 0:
            save_model(model, args.experiment_name, iteration + 1)

    logging.info("We have finished training the model!")
    return validation_accs

def outer_maml_step(model, outer_optimizer, dataloader, device, args, split):
    '''
    Performs the outer training step for MAML.
    '''
    # Sample train and test
    tr_batch, ts_batch, _ = dataloader.sample_task(meta_batch_size=args.meta_batch_size, k_train=args.num_support,
                                                   k_test=args.num_query, context_len=args.context_len,
                                                   test_prefix_len=args.test_prefix_len, split=split)
    tr_batch, ts_batch = tr_batch.to(device), ts_batch.to(device)

    # Initialize loss and accuracy lists
    losses = [0 for _ in range(args.num_inner_updates + 1)]
    accuracies = [0 for _ in range(args.num_inner_updates + 1)]

    # We keep track of the model's state dict, which will later be modified
    for param in model.parameters():
        print(param.name)
    assert False
    state_dict = model.state_dict()
    copied_state_dict = {name: val.clone() for name, val in state_dict.items()}

    # Iterate over each task
    for task_num in range(args.meta_batch_size):
        task_tr, task_ts = tr_batch[task_num], ts_batch[task_num]
        inner_maml_step(model, copied_state_dict, args, task_tr, task_ts, losses, accuracies)

    # Perform gradient update
    if split == "train":
        loss = losses[-1] / args.meta_batch_size
        outer_optimizer.zero_grad()
        loss.backward()
        outer_optimizer.step()

    return accuracies[-1] / args.meta_batch_size

def inner_maml_step(model, state_dict, args, task_tr, task_ts, losses, accuracies):
    '''
    Performs the inner training step for MAML for a particular task. Heavily influenced
    by https://github.com/dragen1860/MAML-Pytorch/blob/master/meta.py.
    '''

    # Compute the performance before any updates
    with torch.no_grad():
        # There likely will be an off-by-one error here?
        logits = model.forward_with_state_dict(task_ts, state_dict)
        losses[0] += utils.compute_loss(task_ts, logits)

    # Perform the inner loop updates
    for iteration in range(args.num_inner_updates):
        # Update on training data
        train_logits = model.forward_with_params(task_tr, copied_params)
        tr_loss = utils.compute_loss(task_tr, train_logits)
        grad = torch.autograd.grad(tr_loss, copied_params)
        copied_params = list(map(lambda p: p[1] - args.inner_lr * p[0], zip(grad, copied_params)))

        # Compute the test losses
        logits = model.forward_with_params(task_ts, copied_params)
        losses[iteration + 1] += utils.compute_loss(task_ts, logits)

if __name__ == '__main__':
    # Get the training arguments
    args = get_arguments()

    # Initialize experiment folders
    utils.initialize_experiment(args.experiment_name, args.log_name, args.seed)

    # Initialize the model
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = initialize_model(args.experiment_name, args.model_type,
                             args.load_from_iteration, device, args.embed_dim, args.hidden_dim)

    # Initialize the dataset
    # TO-DO: Enable sampling multiple tasks and sampling from train, val or test specically 
    dataloader = TaskHandler(num_threads=args.num_workers)

    # Train the model using MAML
    validation_accs = train(model, dataloader, device, args)

    # Save/visualize results: TO-DO
