import numpy as np


def preprocess(text):
    text = text.lower()
    words = text.split(" ")
    print(words)

    word_to_id = {}
    id_to_word = {}

    # 添加id
    for word in words:
        if word not in word_to_id:
            id = len(word_to_id)
            word_to_id[word] = id
            id_to_word[id] = word

    print(word_to_id)
    print(id_to_word)

    corpus = [word_to_id[w] for w in words]
    print(corpus)
    corpus = np.array(corpus)
    return corpus, word_to_id, id_to_word


# 生成共现矩阵
def create_co_matrix(corpus, vocab_size, window_size=1):
    '''生成共现矩阵

    :param corpus: 语料库（单词ID列表）
    :param vocab_size:词汇个数
    :param window_size:窗口大小（当窗口大小为1时，左右各1个单词为上下文）
    :return: 共现矩阵
    '''
    corpus_size = len(corpus)
    co_matrix = np.zeros((vocab_size, vocab_size), dtype=np.int32)

    for idx, word_id in enumerate(corpus):
        for i in range(1, window_size + 1):
            left_idx = idx - i
            right_idx = idx + i

            if left_idx >= 0:
                left_word_id = corpus[left_idx]
                co_matrix[word_id, left_word_id] += 1

            if right_idx < corpus_size:
                right_word_id = corpus[right_idx]
                co_matrix[word_id, right_word_id] += 1

    return co_matrix


# 计算余弦相似度
def cos_similarity(x, y, eps=1e-8):
    '''计算余弦相似度

    :param x: 向量
    :param y: 向量
    :param eps: 用于防止“除数为0”的微小值
    :return:
    '''
    nx = x / (np.sqrt(np.sum(x ** 2)) + eps)
    ny = y / (np.sqrt(np.sum(y ** 2)) + eps)
    return np.dot(nx, ny)


# 相似单词的查找
def most_similar(query, word_to_id, id_to_word, word_matrix, top=5):
    '''相似单词的查找

    :param query: 查询词
    :param word_to_id: 从单词到单词ID的字典
    :param id_to_word: 从单词ID到单词的字典
    :param word_matrix: 汇总了单词向量的矩阵，假定保存了与各行对应的单词向量
    :param top: 显示到前几位
    '''
    if query not in word_to_id:
        print('%s is not found' % query)
        return

    print('\n[query] ' + query)
    query_id = word_to_id[query]
    query_vec = word_matrix[query_id]

    vocab_size = len(id_to_word)

    similarity = np.zeros(vocab_size)
    for i in range(vocab_size):
        similarity[i] = cos_similarity(word_matrix[i], query_vec)

    count = 0
    # 排序
    for i in (-1 * similarity).argsort():
        if id_to_word[i] == query:
            continue
        print(' %s: %s' % (id_to_word[i], similarity[i]))

        count += 1
        if count >= top:
            return
