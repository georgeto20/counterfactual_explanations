from asdl.lang.babyai.babyai_transition_system import BabyAITransitionSystem
from asdl.transition_system import *
from asdl.asdl import ASDLGrammar
from components.action_info import get_action_infos
from components.dataset import Example
from model.parser import *
from common.utils import init_arg_parser
from verifier import ObjDesc, GoToInstr, OpenInstr, PickupInstr, PutNextInstr, AndInstr, BeforeInstr, AfterInstr
import random
import numpy as np
import pickle
import argparse
import nltk
import spacy
import copy
import pandas as pd
from string import punctuation
import matplotlib.pyplot as plt

def init_arg_parser():
    arg_parser = argparse.ArgumentParser()

    #### General configuration ####
    arg_parser.add_argument('--seed', default=0, type=int, help='Random seed')
    arg_parser.add_argument('--cuda', action='store_true', default=False, help='Use gpu')
    arg_parser.add_argument('--lang', choices=['python', 'lambda_dcs', 'wikisql', 'prolog', 'python3'], default='python',
                            help='[Deprecated] language to parse. Deprecated, use --transition_system and --parser instead')
    arg_parser.add_argument('--asdl_file', type=str, help='Path to ASDL grammar specification')

    #### Modularized configuration ####
    arg_parser.add_argument('--parser', type=str, default='default_parser', required=False, help='name of parser class to load')
    arg_parser.add_argument('--transition_system', type=str, default='python2', required=False, help='name of transition system to use')
    arg_parser.add_argument('--evaluator', type=str, default='default_evaluator', required=False, help='name of evaluator class to use')

    #### Model configuration ####
    arg_parser.add_argument('--lstm', choices=['lstm'], default='lstm', help='Type of LSTM used, currently only standard LSTM cell is supported')

    # Embedding sizes
    arg_parser.add_argument('--embed_size', default=128, type=int, help='Size of word embeddings')
    arg_parser.add_argument('--action_embed_size', default=128, type=int, help='Size of ApplyRule/GenToken action embeddings')
    arg_parser.add_argument('--field_embed_size', default=64, type=int, help='Embedding size of ASDL fields')
    arg_parser.add_argument('--type_embed_size', default=64, type=int, help='Embeddings ASDL types')

    # Hidden sizes
    arg_parser.add_argument('--hidden_size', default=256, type=int, help='Size of LSTM hidden states')
    arg_parser.add_argument('--ptrnet_hidden_dim', default=32, type=int, help='Hidden dimension used in pointer network')
    arg_parser.add_argument('--att_vec_size', default=256, type=int, help='size of attentional vector')

    # readout layer
    arg_parser.add_argument('--no_query_vec_to_action_map', default=False, action='store_true',
                            help='Do not use additional linear layer to transform the attentional vector for computing action probabilities')
    arg_parser.add_argument('--readout', default='identity', choices=['identity', 'non_linear'],
                            help='Type of activation if using additional linear layer')
    arg_parser.add_argument('--query_vec_to_action_diff_map', default=False, action='store_true',
                            help='Use different linear mapping ')

    # supervised attention
    arg_parser.add_argument('--sup_attention', default=False, action='store_true', help='Use supervised attention')

    # parent information switch for decoder LSTM
    arg_parser.add_argument('--no_parent_production_embed', default=False, action='store_true',
                            help='Do not use embedding of parent ASDL production to update decoder LSTM state')
    arg_parser.add_argument('--no_parent_field_embed', default=False, action='store_true',
                            help='Do not use embedding of parent field to update decoder LSTM state')
    arg_parser.add_argument('--no_parent_field_type_embed', default=False, action='store_true',
                            help='Do not use embedding of the ASDL type of parent field to update decoder LSTM state')
    arg_parser.add_argument('--no_parent_state', default=False, action='store_true',
                            help='Do not use the parent hidden state to update decoder LSTM state')

    arg_parser.add_argument('--no_input_feed', default=False, action='store_true', help='Do not use input feeding in decoder LSTM')
    arg_parser.add_argument('--no_copy', default=False, action='store_true', help='Do not use copy mechanism')

    # Model configuration parameters specific for wikisql
    arg_parser.add_argument('--column_att', choices=['dot_prod', 'affine'], default='affine', help='How to perform attention over table columns')

    #### Training ####
    arg_parser.add_argument('--vocab', type=str, help='Path of the serialized vocabulary')
    arg_parser.add_argument('--glove_embed_path', default=None, type=str, help='Path to pretrained Glove mebedding')

    arg_parser.add_argument('--batch_size', default=10, type=int, help='Batch size')
    arg_parser.add_argument('--dropout', default=0., type=float, help='Dropout rate')
    arg_parser.add_argument('--word_dropout', default=0., type=float, help='Word dropout rate')
    arg_parser.add_argument('--decoder_word_dropout', default=0.3, type=float, help='Word dropout rate on decoder')
    arg_parser.add_argument('--primitive_token_label_smoothing', default=0.0, type=float,
                            help='Apply label smoothing when predicting primitive tokens')
    arg_parser.add_argument('--src_token_label_smoothing', default=0.0, type=float,
                            help='Apply label smoothing in reconstruction model when predicting source tokens')

    arg_parser.add_argument('--negative_sample_type', default='best', type=str, choices=['best', 'sample', 'all'])

    # training schedule details
    arg_parser.add_argument('--valid_metric', default='acc', choices=['acc'],
                            help='Metric used for validation')
    arg_parser.add_argument('--valid_every_epoch', default=1, type=int, help='Perform validation every x epoch')
    arg_parser.add_argument('--log_every', default=10, type=int, help='Log training statistics every n iterations')

    arg_parser.add_argument('--save_to', default='model', type=str, help='Save trained model to')
    arg_parser.add_argument('--save_all_models', default=False, action='store_true', help='Save all intermediate checkpoints')
    arg_parser.add_argument('--patience', default=5, type=int, help='Training patience')
    arg_parser.add_argument('--max_num_trial', default=10, type=int, help='Stop training after x number of trials')
    arg_parser.add_argument('--uniform_init', default=None, type=float,
                            help='If specified, use uniform initialization for all parameters')
    arg_parser.add_argument('--glorot_init', default=False, action='store_true', help='Use glorot initialization')
    arg_parser.add_argument('--clip_grad', default=5., type=float, help='Clip gradients')
    arg_parser.add_argument('--max_epoch', default=-1, type=int, help='Maximum number of training epoches')
    arg_parser.add_argument('--optimizer', default='Adam', type=str, help='optimizer')
    arg_parser.add_argument('--lr', default=0.001, type=float, help='Learning rate')
    arg_parser.add_argument('--lr_decay', default=0.5, type=float,
                            help='decay learning rate if the validation performance drops')
    arg_parser.add_argument('--lr_decay_after_epoch', default=0, type=int, help='Decay learning rate after x epoch')
    arg_parser.add_argument('--decay_lr_every_epoch', action='store_true', default=False, help='force to decay learning rate after each epoch')
    arg_parser.add_argument('--reset_optimizer', action='store_true', default=False, help='Whether to reset optimizer when loading the best checkpoint')
    arg_parser.add_argument('--verbose', action='store_true', default=False, help='Verbose mode')
    arg_parser.add_argument('--eval_top_pred_only', action='store_true', default=False,
                            help='Only evaluate the top prediction in validation')

    #### decoding/validation/testing ####
    arg_parser.add_argument('--load_model', default=None, type=str, help='Load a pre-trained model')
    arg_parser.add_argument('--beam_size', default=5, type=int, help='Beam size for beam search')
    arg_parser.add_argument('--decode_max_time_step', default=100, type=int, help='Maximum number of time steps used '
                                                                                  'in decoding and sampling')
    arg_parser.add_argument('--sample_size', default=5, type=int, help='Sample size')
    arg_parser.add_argument('--save_decode_to', default=None, type=str, help='Save decoding results to file')

    #### reranking ####
    arg_parser.add_argument('--train_decode_file', default=None, type=str, help='Decoding results on training set')
    arg_parser.add_argument('--test_decode_file', default=None, type=str, help='Decoding results on test set')
    arg_parser.add_argument('--dev_decode_file', default=None, type=str, help='Decoding results on dev set')
    arg_parser.add_argument('--metric', default='accuracy', choices=['bleu', 'accuracy'])
    arg_parser.add_argument('--num_workers', default=1, type=int, help='number of multiprocess workers')

    #### self-training ####
    arg_parser.add_argument('--load_decode_results', default=None, type=str)
    arg_parser.add_argument('--unsup_loss_weight', default=1., type=float, help='loss of unsupervised learning weight')

    #### interactive mode ####
    arg_parser.add_argument('--example_preprocessor', default=None, type=str, help='name of the class that is used to pre-process raw input examples')

    #### dataset specific config ####
    arg_parser.add_argument('--sql_db_file', default=None, type=str, help='path to WikiSQL database file for evaluation (SQLite)')

    return arg_parser

def init_config():
    args = arg_parser.parse_args()

    # seed the RNG
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)
    np.random.seed(int(args.seed * 13 / 7))

    return args

def remove_punctuation(text):
    if type(text) == float:
        return ''
    for punct in punctuation:
        text = text.replace(punct, '')
    return text.lower()

if __name__ == '__main__':
    arg_parser = init_arg_parser()
    args = init_config()
    asdl_text = open(args.asdl_file).read()
    grammar = ASDLGrammar.from_text(asdl_text)
    transition_system = BabyAITransitionSystem(grammar)
    parser_cls = Registrable.by_name(args.parser)
    parser = parser_cls.load(model_path=args.load_model, cuda=args.cuda)
    instrs = [pickle.load(open('datasets/babyai/babyai_instructions' + str(i) + '.pkl', 'rb')) for i in range(17)]
    print(len([sentence for l in instrs for sentence in l]))
    print(len(instrs[0]))
    print(len(instrs[1]))
    babyai_asts = [[transition_system.instructions_to_babyai_ast(instruction) for instruction in instrs[i]] for i in range(len(instrs))]
    print(len([tree for l in babyai_asts for tree in l]))
    print(len(babyai_asts[0]))
    print(len(babyai_asts[1]))
    results = {'No Training': [], 'Training': [], 'GPT-2': [], 'No Demo': [], 'Our Tool': []}
    for approach in results.keys():
        csv = pd.read_csv('datasets/babyai/Baby AI Instructions ' + approach + '.csv')
        if approach == 'Training':
            instructions = [csv[column].apply(remove_punctuation).values for column in ['Q5', 'Q6', 'Q7', 'Q8', 'Q10', 'Q11', 'Q12', 'Q13', 'Q33', 'Q34', 'Q35', 'Q36', 'Q38', 'Q39', 'Q40', 'Q41', 'Q42']]
        else:
            instructions = [csv[column].apply(remove_punctuation).values for column in ['Q5', 'Q6', 'Q7', 'Q8', 'Q10', 'Q11', 'Q12', 'Q13', 'Q18', 'Q19', 'Q20', 'Q21', 'Q23', 'Q24', 'Q25', 'Q26', 'Q27']]
        for i in range(len(instructions)):
            count = 0
            for j in range(len(instructions[i]) - 50, len(instructions[i])):
                if instructions[i][j]:
                    trees = parser.parse(nltk.word_tokenize(instructions[i][j]))
                    babyai_tree = transition_system.asdl_ast_to_babyai_ast(trees[0].tree)
                    for k in range(len(babyai_asts[i])):
                        if babyai_asts[i][k] == babyai_tree:
                            count += 1
                            break
            results[approach].append(count / 50)
    for i in range(len(results['No Training'])):
        results['Training'][i] /= results['No Training'][i]
        results['GPT-2'][i] /= results['No Training'][i]
        results['No Demo'][i] /= results['No Training'][i]
        results['Our Tool'][i] /= results['No Training'][i]
        results['No Training'][i] /= results['No Training'][i]
    print('\t\tNo training:\tTraining:\tGPT-2:\t\tNo demo:\tOur tool:')
    for i in range(9):
        print('Task ' + str(i+1) + ':\t\t' + str(results['No Training'][i]) + '\t\t' + str(results['Training'][i]) + '\t\t' + str(results['GPT-2'][i]) + '\t\t' + str(results['No Demo'][i]) + '\t\t' + str(results['Our Tool'][i]))
    for i in range(9, 17):
        print('Task ' + str(i+1) + ':\t' + str(results['No Training'][i]) + '\t\t' + str(results['Training'][i]) + '\t\t' + str(results['GPT-2'][i]) + '\t\t' + str(results['No Demo'][i]) + '\t\t' + str(results['Our Tool'][i]))
    results['Task'] = [i for i in range(1, 18)]
    df = pd.DataFrame.from_dict(results)
    #df.to_csv('datasets/babyai/final_results.csv', index=False)
    #df.plot(x='Task', y=['No Training', 'Training', 'GPT-2', 'No Demo', 'Our Tool'], kind='bar')
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot('Task', 'No Training', data=df, label='No Training', marker='o', color='b')
    #ax.plot('Task', 'Training', data=df, label='Training', marker='o', color='g')
    ax.plot('Task', 'GPT-2', data=df, label='GPT-2', marker='o', color='k')
    ax.plot('Task', 'No Demo', data=df, label='No Demo', marker='o', color='y')
    ax.plot('Task', 'Our Tool', data=df, label='Our Tool', marker='o', color='r')
    ax.set_xticks([i for i in range(1, 18)])
    ax.set_yticks([i for i in range(1, 13)])
    plt.title('% of Matching tranX ASTs per Task')
    plt.xticks(rotation='horizontal')
    plt.ylabel('% of matching tranX ASTs')
    plt.legend()
    plt.show()