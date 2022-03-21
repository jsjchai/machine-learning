import random
import spacy
from spacy.tokens import Span, DocBin

nlp = spacy.blank("zh")

doc1 = nlp("iPhone X就要来了")
doc1.ents = [Span(doc1, 0, 8, label="GADGET")]

doc2 = nlp("我急需一部新手机，给点建议吧！")
docs = [doc1, doc2]

random.shuffle(docs)
train_docs = docs[:len(docs) // 2]
dev_docs = docs[len(docs) // 2:]

# 创建和保存一系列的训练文档
train_docbin = DocBin(docs=train_docs)
train_docbin.to_disk("./train.spacy")
# 创建和保存一系列的测试文档
dev_docbin = DocBin(docs=dev_docs)
dev_docbin.to_disk("./dev.spacy")
