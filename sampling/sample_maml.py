# File: sample_maml
# -----------------
# Use a pre-trained MAML model to generate music predictions.

import argparse
import torch
import torch.nn.functional as F
import numpy as np
import logging
import time
import higher
import glob
import os
import pickle
from tqdm import tqdm

from collections import OrderedDict
import json

import utils
import constants
import vis_utils
from dataset.task_dataset import TaskHandler
from dataset.data_utils import decode
from models.model_utils import initialize_model, save_model


def get_arguments():
    '''
    Uses argparse to get the arguments for the experiment
    '''
    
    parser = argparse.ArgumentParser(description="Few-shot music generation with MAML")

    # Optimization arguments
    parser.add_argument("--num_inner_updates", type=int, default=constants.NUM_INNER_UPDATES,
                        help="Number of inner updates MAML will take in the inner loop")
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
    
    # Data generation args
    parser.add_argument("--condition_len", type=int, default=constants.CONDITION_LENGTH,
                        help="The number of condition sequences provided to the model prior to generation.")
    parser.add_argument("--generation_len", type=int, default=constants.GENERATION_LENGTH,
                        help="The length of the de-novo generation (total sequence will be context + generation length")
    parser.add_argument("--temperature", type=float, default=constants.TEMPERATURE,
                        help="Temperature for sampling from the softmax.")
    
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

def generate(model, dataloader, device, args, split):
    '''
    Generate sequences from the model. Operates in a very similar fashion to the outer loop of MAML,
    except we do not update the outer_optimizer. These arguments were left for ease of adaptation.
    '''

    model.train()

    # Sample train and test
    tr_batch, ts_batch, genres = dataloader.sample_task(meta_batch_size=args.meta_batch_size, k_train=args.num_support,
                                                   k_test=args.num_query, context_len=args.context_len,
                                                   test_prefix_len=args.test_prefix_len, split="test")

    tr_batch, ts_batch = tr_batch.to(device), ts_batch.to(device)

    # Recall that if we pass in a meta-batch that's too big, it gets minned down to the largest possible value
    actual_meta_batch_size = tr_batch.size()[0]

    inner_opt = torch.optim.SGD(model.parameters(), lr=args.inner_lr)

    # Dicts to store generated and reference sequences by genre
    reference_sequences = {}
    generated_sequences = {}

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
                if args.model_type == "SimpleLSTM":
                    support_logits = support_logits.permute(0, 2, 1)
                elif args.model_type == "SimpleTransformer":
                    support_logits = support_logits.permute(1, 2, 0)

                # And the labels need to be (batch, additional_dims)
                support_labels = support_labels.permute(1, 0)

                support_loss = F.cross_entropy(support_logits, support_labels)
                diffopt.step(support_loss)

            # After that, let's generate some samples using the query set as condition
            reference_seq = task_ts
            generated_seq = task_ts[:, :args.condition_len]

            with torch.no_grad():
                for i in range(args.generation_len):

                    # In order to have the transformer not complain, we pass only the context_len - 1 last
                    # tokens of the generated output into the model
                    logits = fnet.forward(generated_seq)

                    # Transformer outputs logits as (batch, seq_len, hidden), so we permute it
                    # to match the expected (seq_len, batch, hidden_)
                    if args.model_type == "SimpleTransformer":
                        logits = logits.permute(1, 0, 2)

                    if args.temperature == 0:
                        pred = torch.argmax(logits[-1, :, :], dim=-1).reshape(-1, 1)

                    else:
                        logits[-1, :, :] /= args.temperature
                        # Note: K-masking not yet implemented
                        log_probs = F.softmax(logits[-1, :, :], dim=-1)
                        pred = torch.multinomial(log_probs, num_samples=1)

                    generated_seq = torch.cat((generated_seq, pred), dim=1)

            reference_sequences[genres[task_num]] = reference_seq
            generated_sequences[genres[task_num]] = generated_seq

    return reference_sequences, generated_sequences


def generate_sequences(model, dataloader, device, args):
    # An artifact of the inner training loop
    with torch.backends.cudnn.flags(enabled=False):

        ref_seqs, gen_seqs = generate(model, dataloader, device, args, "test")

        return ref_seqs, gen_seqs


def create_sample_dir(args):
    BASE_DIR = "./sampling/samples"
    run_id = len(glob.glob(os.path.join(BASE_DIR, "*")))

    # Create a unique directory for this run (e.g. run_0)
    sample_dir = os.path.join(BASE_DIR, "run_{}".format(run_id))
    os.mkdir(sample_dir)

    # Write out args to a json for parameter tracking
    with open(os.path.join(sample_dir, "run_args.json"), "w") as fp:
        json.dump(vars(args), fp, sort_keys=True, indent=4)

    return sample_dir


def process_sequences(ref_seqs_dict, gen_seqs_dict, sample_dir):

    assert all([x == y for (x, y) in zip(ref_seqs_dict.keys(), gen_seqs_dict.keys())]), "Reference and Generated sequence genres do not match"

    # Iterate over genre and song index and write out the decoded midi streams to files
    for genre in ref_seqs_dict:

        ref_seqs = ref_seqs_dict[genre]
        gen_seqs = gen_seqs_dict[genre]

        for song_idx in range(ref_seqs.shape[0]):

            ref = ref_seqs[song_idx, :].cpu().numpy().tolist()
            gen = gen_seqs[song_idx, :].cpu().numpy().tolist()

            ref_stream = decode(ref)
            gen_stream = decode(gen)

            # Write out both the reference and generated sequences
            ref_stream.write('midi', fp=os.path.join(sample_dir, "{}_{}_{}.midi".format(genre, song_idx, "reference")))
            gen_stream.write('midi', fp=os.path.join(sample_dir, "{}_{}_{}.midi".format(genre, song_idx, "generated")))


if __name__ == '__main__':

    # Get the arguments
    args = get_arguments()

    # Initialize the model
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = initialize_model(args.experiment_name, args.model_type,
                             args.load_from_iteration, device, args)
    if args.model_type == "SimpleTransformer":
        model.adaptive_mask = True

    print("Generating sequences with condition length of {} and generation length of {}.".format(args.condition_len, args.generation_len))
    print("Total generated sequence length will be: {}".format(args.condition_len + args.generation_len))
    print("Generation sampling temperature: ", args.temperature)
    print()

    # Initialize the dataset
    # Enable sampling multiple tasks and sampling from train, val or test specically 
    dataloader = TaskHandler(tracks="all-no_drums", num_threads=args.num_workers)

    ref_seqs, gen_seqs = generate_sequences(model, dataloader, device, args)

    example_genre = next(iter(ref_seqs))
    print("Example reference sequence: ", ref_seqs[example_genre][0, :])
    print("Example generated sequence: ", gen_seqs[example_genre][0, :])

    sample_dir = create_sample_dir(args)
    process_sequences(ref_seqs, gen_seqs, sample_dir)
    print("Wrote decoded midi files to {}.".format(sample_dir))

    # Save the raw sequences for reference
    with open(os.path.join(sample_dir, "raw_sequences.pkl"), "wb") as fp:
        pickle.dump({"ref": ref_seqs, "gen": gen_seqs}, fp, protocol=pickle.HIGHEST_PROTOCOL)
