from transformers import AutoModelForSeq2SeqLM

from utils import print_trainable_parameters

model = AutoModelForSeq2SeqLM.from_pretrained('Salesforce/codet5-base')

print(model)

telly_layer = 1

for n, p in model.named_parameters():
    if 'decoder.' in n:
        p.requires_grad = False
    if 'shared.weight' == n:
        p.requires_grad = False
    if n == 'decoder.final_layer_norm.weight':
        p.requires_grad = False
    for j in range(0, telly_layer):
        if f'encoder.block.{j}.' in n:
            p.requires_grad = False
    if '.EncDecAttention.' in n:
        p.requires_grad = True

    print(n, p.requires_grad)

print_trainable_parameters(model)

