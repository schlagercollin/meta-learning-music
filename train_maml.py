# File: train_maml
# ----------------
# Training script for models using MAML.

import random
import argparse
import torch
import torch.nn.functional as F
import numpy as np
import logging
import time
import higher
from tqdm import tqdm

import utils
import constants
import vis_utils
from dataset.task_dataset import TaskHandler
from dataset.maestro_dataset import MaestroDataset
from models.model_utils import initialize_model, save_model, save_entire_model

def get_arguments():
    '''--
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
    parser.add_argument("--num_blocks", type=int, default=constants.NUM_BLOCKS,
                        help="Number of transformer blocks")
    parser.add_argument("--num_heads", type=int, default=constants.NUM_HEADS,
                        help="Number of attention heads")

    # Data loading arguments
    parser.add_argument("--dataset", type=str, default="lakh",
                        help="The type of dataset to train on")
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
    parser.add_argument("--num_test_iterations", type=int, default=constants.TESTING_ITERATIONS,
                        help="How many meta-test steps we wish to perform.")
    parser.add_argument("--only_test", action='store_true',
                        help="If set, we only test model performance. Assumes that a checkpoint is supplied.")
    parser.add_argument("--test_zero_shot", action='store_true',
                        help="If set, we test model performance without an inner loop or finetuning.")

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
    # This appears to be necessary otherwise a _cudnn_rnn_backward RuntimeError is thrown
    # Does this prevent us from using CUDA though?
    with torch.backends.cudnn.flags(enabled=False):
        # Initialize the optimizer
        outer_optimizer = torch.optim.Adam(model.parameters(), lr=args.outer_lr)

        # Initialize the validation loss list
        validation_losses = []

        # Perform the outer updates
        try:
            for iteration in tqdm(range(args.num_train_iterations), desc="Running MAML"):
                # Train step
                avg_loss = outer_maml_step(model, outer_optimizer, dataloader, device, args, "train")

                # Report training accuracy
                if (iteration + 1) % args.report_train_every == 0:
                    logging.info("Average Training Loss for Iteration {}/{}: {}".format(iteration + 1, args.num_train_iterations,
                                                                                avg_loss))

                # Perform validation
                if (iteration + 1) % args.evaluate_every == 0:
                    avg_loss = outer_maml_step(model, outer_optimizer, dataloader, device, args, "val")
                    logging.info("Average Validation Loss for Iteration {}/{}: {}".format(iteration + 1, args.num_train_iterations,
                                                                                    avg_loss))
                    validation_losses.append(avg_loss)

                # Save the model
                if (iteration + 1) % args.save_checkpoint_every == 0:
                    save_model(model, args.experiment_name, iteration + 1)
                    #save_entire_model(model, args.experiment_name, iteration + 1)

        except KeyboardInterrupt:
            logging.info("Keyboard interrupt! Exiting training loop early.")
            save_model(model, args.experiment_name, iteration + 1)
            #save_entire_model(model, args.experiment_name, iteration + 1)
            pass

        logging.info("We have finished training the model!")
        return validation_losses

def outer_maml_step(model, outer_optimizer, dataloader, device, args, split):
    '''
    Performs the outer training step for MAML.
    '''

    model.train()

    # Sample train and test
    if args.dataset == "lakh":
        tr_batch, ts_batch, _ = dataloader.sample_task(meta_batch_size=args.meta_batch_size, k_train=args.num_support,
                                                       k_test=args.num_query, context_len=args.context_len,
                                                       test_prefix_len=args.test_prefix_len, split=split)

    else:
        if split == "train":
            dataloader.train()
        elif split == "val":
            dataloader.val()
        elif split == "test":
            dataloader.test()

        idxs = random.sample(range(len(dataloader)), k=args.meta_batch_size)
        tr_samples, ts_samples = list(zip(*[dataloader[idx] for idx in idxs]))
        tr_batch = torch.stack(tr_samples, dim=0)
        ts_batch = torch.stack(ts_samples, dim=0)


    tr_batch, ts_batch = tr_batch.to(device), ts_batch.to(device)

    # Recall that if we pass in a meta-batch that's too big, it gets minned down to the largest possible value
    actual_meta_batch_size = tr_batch.size()[0]

    inner_opt = torch.optim.SGD(model.parameters(), lr=args.inner_lr)
    query_losses = []
    outer_optimizer.zero_grad()

    # Iterate over each task
    for task_num in range(actual_meta_batch_size):
        task_tr, task_ts = tr_batch[task_num], ts_batch[task_num]

        with higher.innerloop_ctx(model, inner_opt, copy_initial_weights=False) as (fnet, diffopt):
            # Inside the inner loop, do gradient steps on the support set
            for _ in range(args.num_inner_updates):
                support_input, support_labels = task_tr[:, :-1], task_tr[:, 1:]
                support_logits = fnet.forward(support_input)

                # The class dimension needs to go in the middle for the CrossEntropyLoss, and the 
                # necessary permute for this depends on the type of model
                support_logits = support_logits.permute(0, 2, 1)

                # And the labels need to be (batch, additional_dims)
                support_labels = support_labels.permute(1, 0)

                support_loss = F.cross_entropy(support_logits, support_labels)
                diffopt.step(support_loss)

            # After that, calculate the loss (for outer optimization) on the query set
            query_input, query_labels = task_ts[:, :-1], task_ts[:, 1:]
            query_logits = fnet.forward(query_input)

            # The class dimension needs to go in the middle for the CrossEntropyLoss
            query_logits = query_logits.permute(0, 2, 1)

            # And the labels need to be (batch, additional_dims)
            query_labels = query_labels.permute(1, 0)

            query_loss = F.cross_entropy(query_logits, query_labels)
            query_losses.append(query_loss.item())
            query_loss.backward()


    # If we're training, then step the outer optimizer
    if split == "train":
        outer_optimizer.step()

    return np.mean(query_losses)

def test(model, dataloader, device, args):
    '''
    Testing function for MAML.
    '''
    # An artifact of the inner training loop
    # Initialize the optimizer
    with torch.backends.cudnn.flags(enabled=False):    
        outer_optimizer = torch.optim.Adam(model.parameters(), lr=args.outer_lr)

        # Initialize the test loss list
        test_losses = []

        # Perform the meta-test iterations
        for iteration in tqdm(range(args.num_test_iterations), desc="Running MAML"):
            if not args.test_zero_shot: 
                avg_loss = outer_maml_step(model, outer_optimizer, dataloader, device, args, "test")
            else:
                avg_loss = evaluate_zero_shot(model, dataloader, device, args)
            test_losses.append(avg_loss)

        return np.mean(test_losses), np.std(test_losses)

def evaluate_zero_shot(model, dataloader, device, args):
    '''
    Evaluates the model's performance on zero-shot adaptation to test data.
    This is to see how necessary adaptation is.
    '''
    model.eval()

    # Sample test
    if args.dataset == "lakh":
        _, ts_batch, _ = dataloader.sample_task(meta_batch_size=args.meta_batch_size, k_train=args.num_support,
                                                k_test=args.num_query, context_len=args.context_len,
                                                test_prefix_len=args.test_prefix_len, split="test")
    elif args.dataset == "maestro":
        dataloader.test()

        idxs = random.sample(range(len(dataloader)), k=args.meta_batch_size)
        _, ts_samples = list(zip(*[dataloader[idx] for idx in idxs]))
        ts_batch = torch.stack(ts_samples, dim=0)

    B, Q, T = ts_batch.shape
    ts_batch = ts_batch.to(device).view(B*Q, T)

    # Perform predictions on the test batch
    query_input, query_labels = ts_batch[:, :-1], ts_batch[:, 1:]
    query_logits = model.forward(query_input)

    # The class dimension needs to go in the middle for the CrossEntropyLoss
    query_logits = query_logits.permute(0, 2, 1)

    # And the labels need to be (batch, additional_dims)
    query_labels = query_labels.permute(1, 0)
    query_loss = F.cross_entropy(query_logits, query_labels)
    return query_loss.item()

if __name__ == '__main__':
    # Get the training arguments
    args = get_arguments()

    # Initialize experiment folders
    utils.initialize_experiment(args.experiment_name, args.log_name, args.seed, args)

    # Initialize the model
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = initialize_model(args.experiment_name, args.model_type,
                             args.load_from_iteration, device, args, load_whole_object=False)

    # Initialize the dataset
    # Enable sampling multiple tasks and sampling from train, val or test specically 
    if args.dataset == "lakh":
        dataloader = TaskHandler(tracks="all-no_drums", num_threads=args.num_workers)
    elif args.dataset == "maestro":
        dataloader = MaestroDataset(context_len=args.context_len,
                                    k_train=args.num_support,
                                    k_test=args.num_query,
                                    meta=True)

    if not args.only_test:
        # Train the model using MAML
        validation_losses = train(model, dataloader, device, args)

        # Visualize validation losses
        vis_utils.plot_losses(validation_losses, args.evaluate_every, title="Validation Losses",
                              xlabel="Iterations", ylabel="Loss", folder=utils.get_plot_folder(args.experiment_name),
                              name="validation_losses.png")

    mean_test_loss, test_loss_std = test(model, dataloader, device, args)
    logging.info("The mean test loss was {} with standard deviation {}".format(mean_test_loss,
                                                                                   test_loss_std))

