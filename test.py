from transformers import AutoTokenizer, AutoModel
import torch
# Sentences we want sentence embeddings for
sentences = ["样例数据-1", "样例数据-2", "样例数据-2", "样例数据-2", "样例数据-2"]

# Load model from HuggingFace Hub
tokenizer = AutoTokenizer.from_pretrained('BAAI/bge-large-zh-v1.5')
model = AutoModel.from_pretrained('BAAI/bge-large-zh-v1.5')
model.eval()

# Tokenize sentences
encoded_input = tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')
# for s2p(short query to long passage) retrieval task, add an instruction to query (not add instruction for passages)
# encoded_input = tokenizer([instruction + q for q in queries], padding=True, truncation=True, return_tensors='pt')

batch_size = 2
anchor_embeddings = []

for i in range(0, len(encoded_input['input_ids']), batch_size):
    batch_inputs = {k: v[i:i + batch_size] for k, v in encoded_input.items()}
    with torch.no_grad():
        batch_embeddings = torch.nn.functional.normalize(model(**batch_inputs)[0][:, 0], p=2, dim=1)
    anchor_embeddings.append(batch_embeddings.cpu())

# Compute token embeddings
with torch.no_grad():
    model_output = model(**encoded_input)
    # Perform pooling. In this case, cls pooling.
    sentence_embeddings = model_output[0][:, 0]
# normalize embeddings
sentence_embeddings = torch.nn.functional.normalize(sentence_embeddings, p=2, dim=1)
print("Sentence embeddings:", anchor_embeddings)