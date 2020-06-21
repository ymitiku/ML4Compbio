import re
from collections import Counter
import numpy as np
import argparse
import os
import tensorflow as tf
import tensorboard as tb
tf.io.gfile = tb.compat.tensorflow_stub.io.gfile
from torch.utils.tensorboard import SummaryWriter


from tensorboard.plugins import projector

def read_fasta_file(path):
    """
    Gets genome sequence saved in a file with FASTA format. It is assumed that file is in the following format.

    >SEQUENCE_1
    MTEITAAMVKELRESTGAGMMDCKNALSETNGDFDKAVQLLREKGLGKAAKKADRLAAEG
    LVSVKVSDDFTIAAMRPSYLSYEDLDMTFVENEYKALVAELEKENEERRRLKDPNKPEHK
    IPQFASRKQLSDAILKEAEEKIKEELKAQGKPEKIWDNIIPGKMNSFIADNSQLDSKLTL
    MGQFYVMDDKKTVEQVIAEKEKEFGGKIKIVEFICFEVGEGLEKKTEDFAAEVAAQL
    >SEQUENCE_2
    SATVSEINSETDFVAKNDQFIALTKDTTAHIQSNSLQSVEELHSSTINGVKFEEYLKSQI
    ATIGENLVVRRFATLKAGANGVVNGYIHTNGRVGVVIAAACDSAEVASKSRDLLRQICMH

    Paramters
    ---------
    Path : str
        Path to file containing genome sequence.
    Returns
    -------
    dict
        A key-value pair where the keys are sequence names and values are genome sequences
    """
    with open(path) as data_file:
        output = {}
        sequence_name = None
        for line in data_file.readlines():
            if line.startswith(">"):
                sequence_name = line[1:].strip()
            else:
                output.setdefault(sequence_name, "")
                line = "".join(re.findall("[acgtACGT]+", line))

                output[sequence_name]+=line.upper()
        return output

def get_kmers(seq, k):
    """
    Gets kmers for the given sequence.

    Parameters
    ----------
    seq : str
        Genomic sequence
    k : int
        Length of the kmers to return
    Returns
    -------
    list
        List of kmers of length k.
    """

    return [seq[i:i+k] for i in range(len(seq)-k+1)]

def get_sequences_kmers_counts(sequences, k, index2animal):
    
    kmers_counts =[]
    kmers_vocabulary = set()

    for i in range(len(index2animal)):
        seq_kmers = get_kmers(sequences[index2animal[i]], k)
        kmers_counts.append(Counter(seq_kmers))
        kmers_vocabulary.update(seq_kmers)
    return kmers_counts, kmers_vocabulary

def build_vector_representation(sequences, k):
    animals = list(sequences.keys())
    index2animal = {i:animals[i] for i in range(len(animals))}

    kmers_counts, kmers_vocabulary = get_sequences_kmers_counts(sequences, k, index2animal)
    indexed_voc = list(kmers_vocabulary)

    kmer2index = {indexed_voc[i]:i for i in range(len(indexed_voc))}

    

    representation = np.zeros((len(sequences), len(indexed_voc)))
    for i in range(len(sequences)):
        seq_kmer_counts = kmers_counts[i]
        for kmer in seq_kmer_counts:
            representation[i][kmer2index[kmer]] = seq_kmer_counts[kmer]
    return representation, index2animal

    

    

def main(args):
    path = args.path
    
    sequences = read_fasta_file(path)
    k = args.k
    representation, index2animal = build_vector_representation(sequences, k)
    
    animal2index = {value:key for key, value in index2animal.items()}
    
    log_dir='./logs/'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)


    if args.task == "emb" or args.task=="all":
        metadata_filename = "meta-data-{}mer.csv".format(k)
        meta_data = []
        
        for i in range(len(index2animal)):
            meta_data.append(index2animal[i])

        writer = SummaryWriter()
        writer.add_embedding(representation, meta_data)
        writer.close()
    if args.task == "sim" or args.task == "all":
        animal = args.animal
        animal_index = animal2index[animal]
        similarity_matrix = np.zeros((len(index2animal), len(index2animal)))
        for i in range(len(index2animal)):
            for j in range(i+1, len(index2animal)):
                similarity = np.dot(representation[i],  representation[j])/(np.linalg.norm(representation[i])* np.linalg.norm(representation[j]))
            
                similarity_matrix[i][j] = similarity
                similarity_matrix[j][i] = similarity
        most_sim_index = similarity_matrix[animal_index, :].argmax()
        print("The most similar to {} genome is {} genome".format(animal, index2animal[most_sim_index]))
    
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--path", required=True, help="Path to data file")
    parser.add_argument("-t", "--task", default="emb", choices=["sim", "emb", "all"], help="Which task to execute. This choices are `sim` for similarity, `emb` for embedding generation")
    parser.add_argument("-a", "--animal", choices=["human", "chimp", "mouse", "rat", "dog", "cow", "armadillo", "elephant"], help="Animal name where other animal that is most similar should be searched")
    parser.add_argument("-k", default=3, type = int, help="Value of k for kmer")
    args = parser.parse_args()
    main(args)
    