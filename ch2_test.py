# input_text = "나는 최근 파리 여행을 다녀왔다"
# input_text_list = input_text.split()
# print("input_text_list: ", input_text_list)
#
# str2idx = {word:idx for idx, word in enumerate(input_text_list)}
# idx2str = {idx:word for idx, word in enumerate(input_text_list)}
# print("str2idx: ", str2idx)
# print("idx2str: ", idx2str)
#
# input_ids = [str2idx[word] for word in input_text_list]
# print("input_ids: ", input_ids)


import torch
import torch.nn as nn

# 가정: str2idx와 input_ids가 정의되어 있어야 합니다.
# 예시 데이터를 임의로 생성했습니다.
str2idx = {"hello": 0, "world": 1, "pytorch": 2, "embedding": 3, "test": 4}
input_ids = [0, 1, 2, 3, 4]

embedding_dim = 16
embed_layer = nn.Embedding(len(str2idx), embedding_dim)

# (5, 16) 형태로 임베딩 생성
input_embeddings = embed_layer(torch.tensor(input_ids))

print(input_embeddings)

# (1, 5, 16)으로 차원 확장 (Batch size 추가)
input_embeddings = input_embeddings.unsqueeze(0)

print(input_embeddings.shape)