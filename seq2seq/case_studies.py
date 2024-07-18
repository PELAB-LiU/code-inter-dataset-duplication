from analyze_results import read_data
from evaluation_metrics import get_normalization, nltk_sentence_bleu

from datasets import load_dataset

data_folder_biased = '/data/ja_models/final_models/code2text_tlc/seed_1/random_biased/best_checkpoint/'
# data_folder_unbiased = '/data/ja_models/final_models/code2text/seed_12/random_unbiased/best_checkpoint/'

normalization = get_normalization('code2text', 'java')
references_dup_biased, predictions_dup_biased = read_data(data_folder_biased, 'dup', normalization)
dataset = load_dataset('antolin/tlc_interduplication').filter(lambda x: x["is_duplicated"])
# references_dup_unbiased, predictions_dup_unbiased = read_data(data_folder_unbiased, 'dup', normalization)

memorized = []
cont = 0
for i in range(len(references_dup_biased)):
    bleu_biased = nltk_sentence_bleu(predictions_dup_biased[i], references_dup_biased[i])
    if bleu_biased > 0.6:
        memorized.append(i)
        print('remembered')
        print('prediction:')
        print(' '.join(predictions_dup_biased[i]))
        print('ground truth')
        print(dataset["test"][i]['nl'])
        print('snippet')
        print(dataset["test"][i]['snippet'])
        print('---'*50)
        cont += 1
    if cont > 8:
        break

half_memorized = []
cont = 0
for i in range(len(references_dup_biased)):
    bleu_biased = nltk_sentence_bleu(predictions_dup_biased[i], references_dup_biased[i])
    if 0.3 < bleu_biased < 0.6:
        half_memorized.append(i)
        print('partially remembered')
        print('prediction:')
        print(' '.join(predictions_dup_biased[i]))
        print('ground truth')
        print(dataset["test"][i]['nl'])
        print('snippet')
        print(dataset["test"][i]['snippet'])
        print('---' * 50)
        cont += 1
    if cont > 8:
        break

not_memorized = []
cont = 0
for i in range(len(references_dup_biased)):
    bleu_biased = nltk_sentence_bleu(predictions_dup_biased[i], references_dup_biased[i])
    if bleu_biased < 0.3:
        not_memorized.append(i)
        print('not remembered')
        print('prediction:')
        print(' '.join(predictions_dup_biased[i]))
        print('ground truth')
        print(dataset["test"][i]['nl'])
        print('snippet')
        print(dataset["test"][i]['snippet'])
        print('---' * 50)
        cont += 1
    if cont > 8:
        break
