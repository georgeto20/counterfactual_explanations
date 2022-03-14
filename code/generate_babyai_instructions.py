import babyai#.babyai
import gym
import pickle
import sys
import argparse
import json
import os
import pickle
import sys
import numpy as np
#import nltk
from asdl.lang.babyai.babyai_transition_system import BabyAITransitionSystem
from asdl.transition_system import *
from asdl.asdl import ASDLGrammar
from components.action_info import get_action_infos
from components.dataset import Example
from components.vocab import Vocab, VocabEntry
from datasets.babyai.verifier import *

"""
Preprocess generated instructions and
convert them to train, dev, and test set.
"""
def preprocess_dataset(instructions, grammar_file, src_freq=3, code_freq=3, vocab_size=20000, out_dir='data'):
    
    np.random.seed(1234)

    # Read grammar file and create grammar
    asdl_text = open(grammar_file).read()
    grammar = ASDLGrammar.from_text(asdl_text)
    transition_system = BabyAITransitionSystem(grammar)
    examples = []

    # Iterate through each pair of English and Python instructions
    for (english_instruction, python_instruction) in instructions:
        asdl_ast = transition_system.instructions_to_asdl_ast(python_instruction, grammar)
        print(asdl_ast.to_string())
        actions = transition_system.get_actions(asdl_ast)
        english_instructions_tokens = nltk.word_tokenize(english_instruction)
        action_infos = get_action_infos(english_instructions_tokens, actions)
        # Create an Example object of the English instruction,
        # the target AST, and the actions necessary to create it
        example = Example(src_sent=english_instructions_tokens,
                          tgt_actions=action_infos,
                          tgt_code=english_instructions_tokens,
                          tgt_ast=asdl_ast)
        examples.append(example)

    np.random.shuffle(examples)
    # held out 200 examples for development
    dev_examples = examples[:200]
    test_examples = examples[200:400]
    train_examples = examples[400:]

    # Create vocab
    src_vocab = VocabEntry.from_corpus([e.src_sent for e in train_examples], size=vocab_size,
                                       freq_cutoff=src_freq)
    print(src_vocab.word2id)
    primitive_tokens = [map(lambda a: a.action.token,
                            filter(lambda a: isinstance(a.action, GenTokenAction), e.tgt_actions))
                        for e in train_examples]
    primitive_vocab = VocabEntry.from_corpus(primitive_tokens, size=vocab_size, freq_cutoff=code_freq)
    print(primitive_vocab.word2id)
    code_vocab = src_vocab

    vocab = Vocab(source=src_vocab, primitive=primitive_vocab, code=code_vocab)
    print('generated vocabulary %s' % repr(vocab), file=sys.stderr)

    action_lens = [len(e.tgt_actions) for e in train_examples]
    print('Max action len: %d' % max(action_lens), file=sys.stderr)
    print('Avg action len: %d' % np.average(action_lens), file=sys.stderr)
    print('Actions larger than 100: %d' % len(list(filter(lambda x: x > 100, action_lens))), file=sys.stderr)

    # Dump everything to pickle files
    #pickle.dump(train_examples, open(os.path.join(out_dir, 'train.bin'), 'wb'))
    #pickle.dump(dev_examples, open(os.path.join(out_dir, 'dev.bin'), 'wb'))
    #pickle.dump(test_examples, open(os.path.join(out_dir, 'test.bin'), 'wb'))
    #pickle.dump(vocab, open(os.path.join(out_dir, 'vocab.bin'), 'wb'))

"""
Randomly sample and print out instructions from all levels.
"""
def generate_instructions():
    # List of all 19 BabyAI levels
    levels = ["BabyAI-GoToObj-v0", "BabyAI-GoToRedBallGrey-v0", "BabyAI-GoToRedBall-v0", "BabyAI-GoToLocal-v0", "BabyAI-PutNextLocal-v0", "BabyAI-PickupLoc-v0", "BabyAI-GoToObjMaze-v0", "BabyAI-GoTo-v0", "BabyAI-Pickup-v0", "BabyAI-UnblockPickup-v0", "BabyAI-Open-v0", "BabyAI-Unlock-v0", "BabyAI-PutNext-v0", "BabyAI-Synth-v0", "BabyAI-SynthLoc-v0", "BabyAI-GoToSeq-v0", "BabyAI-SynthSeq-v0", "BabyAI-GoToImpUnlock-v0", "BabyAI-BossLevel-v0"]
    # List of generated instructions
    instructions = []
    # List of number of generated instructions per level
    counts = []
    # Iterate over all levels
    for index, level in enumerate(levels):
        sys.stdout.write('Generating instructions for level {} of {}...\r'.format(index + 1, len(levels)))
        sys.stdout.flush()
        # Set up the environment for this level
        env = gym.make(level)
        # You can change this number based on how many instructions you want to generate
        NUMBER_OF_ITERATIONS = 100
        # Reset this environment many times and collect the English instructions
        # and the Python object corresponding to those instructions
        new_instructions = [(env.reset()['mission'], env.instrs) for i in range(NUMBER_OF_ITERATIONS)]
        print("Sample English instructions and Python object for level {}: {} {}".format(level.split('-')[1], new_instructions[0][0], new_instructions[0][1]))
        old_len = len(instructions)
        # Only add those instructions which have not been generated in previous levels
        for (new_english_instruction, new_python_instruction) in new_instructions:
            if new_english_instruction not in [tup[0] for tup in instructions]:
                instructions.append((new_english_instruction, new_python_instruction))
        counts.append(len(instructions) - old_len)
    print("You have generated a total of {} instructions.".format(sum(counts)))
    print("Here is the distribution of generated instructions across all levels:", counts)
    return instructions

def construct_sentence_vector(description, vectors):
    sentence_vector = np.zeros(shape=(vectors.dim,))
    for word in description.split():
        word_vector = vectors.query(word)
        sentence_vector += word_vector
    sentence_vector /= len(description.split())
    return sentence_vector

def choose_best_instruction(user_instruction, known_instructions, constructed_instructions, vectors):
    constructed_user_instruction = construct_sentence_vector(user_instruction, vectors)
    most_similar_instruction = ""
    max_similarity = 0
    for i in range(len(known_instructions)):
        current_similarity = vectors.similarity(constructed_user_instruction, constructed_instructions[i])
        if current_similarity > max_similarity:
            most_similar_instruction = known_instructions[i]
            max_similarity = current_similarity
    return most_similar_instruction

if __name__ == '__main__':
    instructions = generate_instructions()
    #preprocess_dataset(instructions, '../../asdl/lang/babyai/babyai_asdl.txt')