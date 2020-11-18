# File: train_maml
# ----------------
# Training script for models using MAML.

import argparse
import torch
import torch.nn.functional as F
import numpy as np
import logging
import time
import math
import higher
from tqdm import tqdm
from torch.utils.data import DataLoader

import utils
import constants
import vis_utils
from dataset.baseline_dataset import BaselineDataset
from models.model_utils import initialize_model, save_model, save_entire_model

def get_arguments():
    '''
    Uses argparse to get the arguments for the experiment
    '''
    
    parser = argparse.ArgumentParser(description="Few-shot music generation with MAML")

    # Optimization arguments
    parser.add_argument("--lr", type=float, default=constants.BASELINE_LR,
                        help="The learning rate used for the optimization")
    parser.add_argument("--num_epochs", type=int, default=constants.BASELINE_NUM_EPOCHS)

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
    parser.add_argument("--batch_size", type=int, default=constants.BASELINE_BATCH_SIZE,
                        help="Batch size for training")
    parser.add_argument("--num_workers", type=int, default=constants.NUM_WORKERS,
                        help="Number of threads to use in the data loader")
    parser.add_argument("--context_len", type=int, default=constants.CONTEXT_LEN,
                        help="The length of the training snippets")
    
    # Miscellaneous evaluation and checkpointing arguments
    parser.add_argument("--model_type", type=str, default="SimpleLSTM", choices=constants.MODEL_TYPES,
                        help="The name of the model class to be used")
    parser.add_argument("--report_train_every", type=int, default=constants.BASELINE_REPORT_TRAIN_EVERY,
                        help="Report the training accuracy every report_train_every iterations")
    parser.add_argument("--evaluate_every", type=int, default=constants.BASELINE_VAL_EVERY,
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

    # Experiment arguments
    parser.add_argument("--experiment_name", type=str, default="baseline_test",
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
    
    # Set to train mode (train split)
    dataloader.dataset.train()
    
    # Initialize the optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Initialize the validation loss list
    validation_losses = []

    iteration = 0

    try:
        for epoch in range(args.num_epochs):
            with tqdm(dataloader, total=math.ceil(len(dataloader.dataset)/args.batch_size)) as progbar:
                progbar.set_description("[Epoch {}/{}]. Running batches...".format(epoch, args.num_epochs))
                for batch in progbar:
                    # print("Batch starting!")
                    batch = batch.to(device)
                    inputs, labels = batch[:, :-1], batch[:, 1:]

                    # print("Batch split! Input size: ", inputs.shape)

                    # The class dimension needs to go in the middle for the CrossEntropyLoss, and the 
                    # necessary permute for this depends on the type of model
                    logits = model(inputs)
                    if args.model_type == "SimpleLSTM":
                        logits = logits.permute(0, 2, 1)
                    elif args.model_type == "SimpleTransformer":
                        logits = logits.permute(1, 2, 0)

                    # print("Logits computed!")

                    # And the labels need to be (batch, additional_dims)
                    labels = labels.permute(1, 0)

                    loss = F.cross_entropy(logits, labels)
                    progbar.set_postfix(Loss=loss.item())

                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()

                    # print("Loss propagated!")

                    if (iteration + 1) % args.report_train_every == 0:
                        logging.info("Average Training Loss for Iteration {}: {}".format(iteration + 1, loss))

                    if (iteration + 1) % args.evaluate_every == 0:
                        val_loss, _ = validate(model, dataloader, device, args)
                        logging.info("Average Validation Loss for Iteration {}: {}".format(iteration + 1, val_loss))
                        validation_losses.append(val_loss)

                    if (iteration + 1) % args.save_checkpoint_every == 0:
                        save_model(model, args.experiment_name, iteration + 1)
                        #save_entire_model(model, args.experiment_name, iteration + 1)

                    iteration += 1

    except KeyboardInterrupt:
        print("Interrupted training.")
        save_model(model, args.experiment_name, iteration + 1)
        #save_entire_model(model, args.experiment_name, iteration + 1)
        pass


    logging.info("We have finished training the model!")
    return validation_losses


def validate(model, dataloader, device, args):
    # First, set the dataloader's dataset into validation mode
    dataloader.dataset.val()

    # Then, set the model into evaluation mode
    model.eval()

    losses = []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Validating', total=math.ceil(len(dataloader.dataset)/args.batch_size)):
            batch = batch.to(device)
            inputs, labels = batch[:, :-1], batch[:, 1:]

            # The class dimension needs to go in the middle for the CrossEntropyLoss
            logits = model(inputs)
            if args.model_type == "SimpleLSTM":
                logits = logits.permute(0, 2, 1)
            elif args.model_type == "SimpleTransformer":
                logits = logits.permute(1, 2, 0)

            # And the labels need to be (batch, additional_dims)
            labels = labels.permute(1, 0)

            loss = F.cross_entropy(logits, labels)
            losses.append(loss.item())

    # Finally, make sure to reset the dataset / model into training mode
    dataloader.dataset.train()
    model.train()

    return np.mean(losses), np.std(losses)

def test(model, dataloader, device, args):
    # First, set the dataloader's dataset into validation mode
    dataloader.dataset.test()

    # Then, set the model into evaluation mode
    model.eval()

    losses = []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Testing', total=math.ceil(len(dataloader.dataset)/args.batch_size)):
            batch = batch.to(device)
            inputs, labels = batch[:, :-1], batch[:, 1:]

            # The class dimension needs to go in the middle for the CrossEntropyLoss
            logits = model(inputs)
            if args.model_type == "SimpleLSTM":
                logits = logits.permute(0, 2, 1)
            elif args.model_type == "SimpleTransformer":
                logits = logits.permute(1, 2, 0)

            # And the labels need to be (batch, additional_dims)
            labels = labels.permute(1, 0)

            loss = F.cross_entropy(logits, labels)
            losses.append(loss.item())

    # Finally, make sure to reset the dataset / model into training mode
    dataloader.dataset.train()
    model.train()

    return np.mean(losses), np.std(losses)

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
    dataset = BaselineDataset(tracks="all-no_drums", seq_len=args.context_len)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0) # this is important else it hangs (multiprocessing issue?)

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

