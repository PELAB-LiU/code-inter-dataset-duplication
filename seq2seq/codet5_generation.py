from datasets import load_dataset
from tqdm import tqdm
from transformers import RobertaTokenizer, T5ForConditionalGeneration, GenerationConfig


def get_input_output(tokens):
    position = None
    for j, t in enumerate(tokens):
        if t == "def":
            position = j + 1
            break
    return [t if j != position else '<extra_id_0>' for j, t in enumerate(tokens)], tokens[position]


tokenizer = RobertaTokenizer.from_pretrained('Salesforce/codet5-base')
model = T5ForConditionalGeneration.from_pretrained('Salesforce/codet5-base')
# print(tokenizer.additional_special_tokens)

text = "def <extra_id_0> ( a , b ) : if a > b : return a else return b"
input_ids = tokenizer(text, return_tensors="pt").input_ids
print(input_ids)
# simply generate a single sequence
generated_ids = model.generate(input_ids, max_length=20)
print(generated_ids)
print(tokenizer.decode(generated_ids[0], skip_special_tokens=True))
print(tokenizer.decode(generated_ids[0], skip_special_tokens=False))

dataset = load_dataset('antolin/python-150_interduplication')["test"]
generation_config = GenerationConfig(max_length=128,
                                     num_beams=10)
model = model.cuda()
for i in tqdm(range(len(dataset)), desc='Pred loop'):
    code, output = get_input_output(dataset[i]['tokens'])
    code = ' '.join(code)
    print(code)
    print(output)
    truth = output
    break
    input_ids = tokenizer(code, return_tensors="pt", max_length=256, truncation=True).input_ids.cuda()
    summary_ids = model.generate(input_ids=input_ids,
                                 generation_config=generation_config)
    pred = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

    # check the decoding
    print(f'Pred: {pred.strip()} -- Truth: {truth.strip()}')
