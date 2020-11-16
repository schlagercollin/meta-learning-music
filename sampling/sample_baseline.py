# File: sample_baseline
# =====================
# Samples from a standard LSTM baseline and 
# outputs to a folder with generated, reference, and combined
# sequences (per genre of reference sequence).

import argparse
import torch
import torch.nn.functional as F
import numpy as np
import logging
import time
import math
import glob
import os
import higher
import pickle
import json
from tqdm import tqdm
from torch.utils.data import DataLoader

import utils
import constants
import vis_utils
from dataset.baseline_dataset import BaselineDataset
from dataset.data_utils import decode
from models.model_utils import initialize_model, save_model

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

    # Data generation args
    parser.add_argument("--condition_len", type=int, default=constants.CONDITION_LENGTH,
                        help="The number of condition sequences provided to the model prior to generation.")
    parser.add_argument("--generation_len", type=int, default=constants.GENERATION_LENGTH,
                        help="The length of the de-novo generation (total sequence will be context + generation length")
    parser.add_argument("--temperature", type=float, default=constants.TEMPERATURE,
                        help="Temperature for sampling from the softmax.")
    parser.add_argument("--num_batch_gen", type=float, default=32,
                        help="Number of *batches* to generate.")
    
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


def generate(model, dataloader, device, args):
    # First, set the dataloader's dataset into validation mode
    dataloader.dataset.test()

    # Then, set the model into evaluation mode
    model.eval()

    reference_sequences = []
    generated_sequences = []

    num_batches = 0

    with torch.no_grad():
        try:
            for batch in tqdm(dataloader, desc='Generating', total=args.num_batch_gen):
                
                # unpack the auxiliary genre info
                batch, genres = batch

                # proceed as normal
                batch = batch.to(device)

                reference_seq = batch
                generated_seq = batch[:, :args.condition_len]
                for i in range(args.generation_len):

                    logits = model(generated_seq)

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

                reference_sequences.append((reference_seq, genres))
                generated_sequences.append((generated_seq, genres))
                num_batches += 1

                if num_batches > args.num_batch_gen:
                    break
        except KeyboardInterrupt:
            print("Keyboard interrupt. Processing generated sequences...")
            pass


    # Finally, make sure to reset the dataset / model into training mode
    dataloader.dataset.train()
    model.train()

    return reference_sequences, generated_sequences

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

def process_sequences(ref_seqs_list, gen_seqs_list, sample_dir):

    assert len(ref_seqs_list) == len(gen_seqs_list), "Should be same length."

    # Iterate over genre and song index and write out the decoded midi streams to files
    for ref, gen in zip(ref_seqs_list, gen_seqs_list):

        ref, genre = ref
        gen, genre = gen


        just_gen = gen[:, ref.shape[1]:].cpu().numpy().tolist()[0]  # exclude reference tokens 
        ref = ref.cpu().numpy().tolist()[0]
        gen = gen.cpu().numpy().tolist()[0]
        genre = genre[0]
        
        try:
            ref_stream = decode(ref)
            gen_stream = decode(gen)
            just_gen_stream = decode(just_gen)

            song_idx = len(glob.glob(os.path.join(sample_dir, f"{genre}_*all.midi")))

            # Write out both the reference and generated sequences
            ref_stream.write('midi', fp=os.path.join(sample_dir, "{}_{}_{}.midi".format(genre, song_idx, "reference")))
            gen_stream.write('midi', fp=os.path.join(sample_dir, "{}_{}_{}.midi".format(genre, song_idx, "all")))
            just_gen_stream.write('midi', fp=os.path.join(sample_dir, "{}_{}_{}.midi".format(genre, song_idx, "generated")))
        except Exception as e:
            print("STREAM ERROR: ", e)


if __name__ == '__main__':
    # Get the training arguments
    args = get_arguments()

    # Initialize experiment folders
    utils.initialize_experiment(args.experiment_name, args.log_name, args.seed, args)

    # Initialize the model
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = initialize_model(args.experiment_name, args.model_type,
                             args.load_from_iteration, device, args, load_whole_object=True)

    if args.model_type == "SimpleTransformer":                            
        model.adaptive_mask = True

    # Initialize the dataset
    dataset = BaselineDataset(tracks="all-no_drums", seq_len=args.context_len, return_genre=True)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0) # this is important else it hangs (multiprocessing issue?)

    ref_seqs, gen_seqs = generate(model, dataloader, device, args)
    print("Example reference sequence: ", ref_seqs[0][0][0, :])
    print("Example generated sequence: ", gen_seqs[0][0][0, :])

    sample_dir = create_sample_dir(args)
    process_sequences(ref_seqs, gen_seqs, sample_dir)
    print("Wrote decoded midi files to {}.".format(sample_dir))

    # Save the raw sequences for reference
    with open(os.path.join(sample_dir, "raw_sequences.pkl"), "wb") as fp:
        pickle.dump({"ref": ref_seqs, "gen": gen_seqs}, fp, protocol=pickle.HIGHEST_PROTOCOL)