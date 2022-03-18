import nltk
import jieba
import paddle

from nltk import FreqDist

paddle.enable_static()
# 启动paddle模式。 0.40版之后开始支持，早期版本不支持
jieba.enable_paddle()

text = '两肺多发小结节，多为磨玻璃密度，较大者位于左肺上叶前段（SE2 Im60）见磨玻璃结节影，大小为4*4mm，边界较清。左肺上叶前段近胸膜处（SE2 Im36）见部分实性结节影，大小为11*9mm，其内可见空洞，边界清楚，增强未见明显强化。 【气管、支气管】走行如常，管腔未见狭窄及扩张，管壁未见明显增厚。 【食道】走行如常，管壁未见增厚，管腔未见明显扩张及局部偏心性狭窄。 【淋巴结】双肺门、纵隔及腋窝未见明显淋巴结肿大。 【纵隔】前中后纵隔未见明显结节、肿块影。 【心脏大血管】大小、形态、位置及密度如常，心包膜未见明显增厚，心包腔未见明显积液。 【胸壁、胸膜、胸腔】胸壁、胸膜未见增厚、结节及肿块影，两侧胸腔未见积液'
seg_list = jieba.cut(text, cut_all=False)
s = " ".join(seg_list)

s_token = nltk.word_tokenize(s)
s_tagged = nltk.pos_tag(s_token)

# 命名实体识别
s_ner = nltk.chunk.ne_chunk(s_tagged)
print(s_ner)

# 文本相似度（统计方式）
freq_dist = FreqDist(s_token)
# for e in freq_dist.items():
#     print(e)

# 取出n个常用的单词
most_common_words = freq_dist.most_common(10)
print(most_common_words)
