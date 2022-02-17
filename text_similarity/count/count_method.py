from text_similarity.count.util import create_co_matrix, preprocess, cos_similarity, most_similar

text = "颈 4 左侧 椎弓 根区 异常 信号 结合 病史 考虑 转移瘤"

corpus, word_to_id, id_to_word = preprocess(text)
vocab_size = len(word_to_id)
window_size = 2

C = create_co_matrix(corpus, vocab_size, window_size)
print(C)

c0 = C[word_to_id['颈']]
c1 = C[word_to_id['椎弓']]
print(cos_similarity(c0, c1))

print(most_similar('转移瘤', word_to_id, id_to_word, C))
