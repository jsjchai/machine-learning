import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl
from matplotlib.font_manager import FontProperties

mpl.rcParams['axes.unicode_minus'] = False
my_font = FontProperties(fname='/Library/Fonts/Arial Unicode.ttf', size=12)

from common.util import create_co_matrix, preprocess, cos_similarity, most_similar, ppmi

text = "颈 4 左侧 椎弓 根区 异常 信号 结合 病史 考虑 转移瘤"

corpus, word_to_id, id_to_word = preprocess(text)
vocab_size = len(word_to_id)
window_size = 3

C = create_co_matrix(corpus, vocab_size, window_size)
print(C)

c0 = C[word_to_id['颈']]
c1 = C[word_to_id['椎弓']]
print(cos_similarity(c0, c1))

print(most_similar('转移瘤', word_to_id, id_to_word, C, 8))

# 正的点互信息
M = ppmi(C, False)

np.set_printoptions(precision=3)
print(M)

print("-" * 100)

# SVD 降维
U, S, V = np.linalg.svd(M)
print(U[0])
print(S[0])
print(V[0])

for word, word_id in word_to_id.items():
    plt.annotate(word, (U[word_id, 0], U[word_id, 1]), fontproperties=my_font)

plt.scatter(U[:, 0], U[:, 1], alpha=0.5)
plt.show()
