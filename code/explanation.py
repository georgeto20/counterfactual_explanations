#!/usr/bin/python3
import sys
import os
os.chdir('/usr/lib/cgi-bin/')
import pickle
from sentence_transformers import SentenceTransformer, util
from string import punctuation
import urllib

most_likely = ['go to a blue ball',
 'go to the red key, then go to a red ball, then put a red ball next to the yellow box',
 'open a blue door and open a purple door and go to a red box',
 'go to the blue box, then go to the yellow ball and go to a purple key',
 'pick up the green box, then go to a blue key, then put a green box next to a green ball',
 'go to the grey ball',
 'go to the grey door and open a grey door and go to the red ball',
 'open a grey door',
 'go to a blue box',
 'go to a green key and pick up the yellow box, then go to the purple key',
 'open the purple door, then go to the green ball, then go to the red box',
 'pick up a yellow box and go to a red door',
 'go to the red box and go to a purple key',
 'pick up the grey box and put the grey box next to a green key',
 'open the blue door, then pick up the blue key and go to a red box',
 'go to the yellow door and go to a green box',
 'open the red door and go to the red key and open the purple door']

model = SentenceTransformer('stsb-distilbert-base')

sentences_for_each_task = [pickle.load(open('babyai_sentences' + str(i) + '.pkl', 'rb')) for i in range(17)]

embeddings_for_each_task = pickle.load(open('embeddings_for_each_task_cpu.pkl', 'rb'))

#print(len(embeddings_for_each_task))

sentences_for_all_tasks = [sentence for l in sentences_for_each_task for sentence in l]

embeddings_for_all_tasks = pickle.load(open('embeddings_for_all_tasks_cpu.pkl', 'rb'))

#print(len(embeddings_for_all_tasks))

def explanation(task_id, user_sentence, variant):
  # GPT-2
  if variant == 0:
    return most_likely[task_id]
  else:
    for punct in punctuation:
        user_sentence = user_sentence.replace(punct, '')
    user_sentence = user_sentence.lower()
    embedding = model.encode(user_sentence, convert_to_tensor=True)
    # No-demo
    if variant == 1:
        cosine_scores = util.pytorch_cos_sim(embedding, embeddings_for_all_tasks)
        closest = sentences_for_all_tasks[cosine_scores.argmax(1)]
    # Our tool
    else:
        cosine_scores = util.pytorch_cos_sim(embedding, embeddings_for_each_task[task_id])
        closest = sentences_for_each_task[task_id][cosine_scores.argmax(1)]
    return closest

if __name__ == '__main__':
    #print('\n')
    params = urllib.parse.parse_qs(os.environ['QUERY_STRING'])
    print(params, file=sys.stderr)
    print('The user sentence is: ' + params['user_sentence'][0], file=sys.stderr)
    print('\nexplanation' + params['task_id'][0] + '=' + explanation(int(params['task_id'][0]), params['user_sentence'][0], int(params['variant'][0])))