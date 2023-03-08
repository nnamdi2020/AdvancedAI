from argon2 import hash_password_raw
from hmmlearn import hmm
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np 
import os
import sys
import fileinput # File I/O Library




#Loads the Training Data Transcription File 
def load_data(file):

    with open(file) as f:
        lines = f.readlines()
        #print (lines)
    return lines 


#Vocabulary Set Function
def vocab_set (documents):
    # Create an empty set to store the unique words
    vocabulary_set = set()

    # Iterate through each document and add the words to the vocabulary set
    for document in documents:
        words = document.lower().split()
        vocabulary_set.update(words)

    # Print the vocabulary set
    print(vocabulary_set)


#Transition Matrix Function
def transitionMatrix():


    # Create an empty matrix to store the transition matrix
    #transition_matrix = np.zeros((len(vocabulary_set),len(vocabulary_set)))

    # Define the number of states
    num_states = 4
    
    # Initialize the transition matrix with random values
    transition_matrix = np.random.rand(num_states, num_states)
    #transition_matrix = np.random.rand(n,n)
    #transition_matrix = np.(3,4,5,6,7)

    # Normalize the transition matrix so that each row sums to 1
    transition_matrix /= transition_matrix.sum(axis=1, keepdims=True)
    print ("--------------------------------------------------------")
    print (transition_matrix)
    return transition_matrix


#Observation Matrix Function
def observationMatrix():


    # Create an empty matrix to store the transition matrix
    #transition_matrix = np.zeros((len(vocabulary_set),len(vocabulary_set)))

    # Define the number of states and the number of possible outputs
    num_states = 4
    num_outputs = 27

    # Initialize the observation matrix with random values
    observation_matrix = np.random.rand(num_states, num_outputs)


    # Normalize the transition matrix so that each row sums to 1
    observation_matrix /= observation_matrix.sum(axis=1, keepdims=True)
    print ("--------------------------------------------------------")
    print (observation_matrix)
    return observation_matrix 


    
#Forward Probability Matrix Function
def forwardProb(transition_matrix,observation_matrix):

    # Define the observation sequence and the model parameters
    observation_seq = [0, 1, 0]  # example sequence of observations
    num_states = 3
    num_outputs = 2
    #transition_matrix = np.array([[0.7, 0.2, 0.1], [0.3, 0.5, 0.2], [0.2, 0.3, 0.5]])
    #observation_matrix = np.array([[0.9, 0.1], [0.2, 0.8], [0.6, 0.4]])
    initial_probabilities = np.array([0.6, 0.2, 0.2])

    # Initialize the forward probabilities
    forward_probabilities = np.zeros((len(observation_seq), num_states))
    forward_probabilities[0, :] = initial_probabilities * observation_matrix[:, observation_seq[0]]

    # Compute the forward probabilities for each observation
    for t in range(1, len(observation_seq)):
        for j in range(num_states):
            forward_probabilities[t, j] = observation_matrix[j, observation_seq[t]] * \
            np.sum(forward_probabilities[t-1, :] * transition_matrix[:, j])

    # Compute the total probability of the observation sequence
    total_probability = np.sum(forward_probabilities[-1, :])

def hmmModel ():
    np.random.seed(42)

    model = hmm.GaussianHMM(n_components=3, covariance_type="full")
    model.startprob_ = np.array([0.6, 0.3, 0.1])
    #model.startprob_ = np.array([0.4,  0.2, 0.3, 0.1])
    model.transmat_ = np.array([[0.7, 0.2, 0.1],
                                [0.3, 0.5, 0.2],
                                [0.3, 0.3, 0.4]])

    #model.transmat_ = transition_matrix
    #model.transmat_ = np.array([[3,4,5,6,7],
    model.means_ = np.array([[0.0, 0.0], [3.0, -3.0], [5.0, 10.0]])
    model.covars_ = np.tile(np.identity(2), (3, 1, 1))
    X, Z = model.sample(100)
    print (X,Z)


#Function Calls 
inputfile = load_data('training-data.txt')
train_data = vocab_set (inputfile)
transition_matrix = transitionMatrix()
observation_matrix  = transitionMatrix()
#forwardProb(transition_matrix,observation_matrix)
#forwardProb(t,o)
#hmmModel(transition_matrix)
hmmModel()

# Data Preparation
#train_data = load_data('project1_input.txt') # load training data
#test_data = load_data('project1_input.txt') # load test data
#train_labels = train_data[1]
#test_labels = test_data[1]


# Feature Extraction
#vectorizer = CountVectorizer()
#train_features = vectorizer.fit_transform(train_data[0]).toarray() # extract BoW features
#test_features = vectorizer.transform(test_data[0]).toarray()




