from common.optimizer import Adam
from common.trainer import Trainer
from common.util import preprocess, create_contexts_target, convert_one_hot
from text_similarity.word2vec.simple_cbow import SimpleCBOW

text = 'You say goodbye and I say hello.'

# 词添加id
corpus, word_to_id, id_to_word = preprocess(text)

window_size = 1
hidden_size = 5
max_epoch = 100
batch_size = 3

# 生成上下文和目标词
contexts, target = create_contexts_target(corpus)
print(contexts)
print(target)
vocab_size = len(word_to_id)
target = convert_one_hot(target, vocab_size)
print(target)
contexts = convert_one_hot(contexts, vocab_size)
print(contexts)

model = SimpleCBOW(vocab_size, hidden_size)
optimizer = Adam()
trainer = Trainer(model, optimizer)
trainer.fit(contexts, target, max_epoch, batch_size)
# trainer.plot()

word_vecs = model.word_vecs
for word_id, word in id_to_word.items():
    print(word, word_vecs[word_id])
