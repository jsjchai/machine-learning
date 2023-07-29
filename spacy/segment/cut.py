import spacy_pkuseg
from spacy.lang.zh import Chinese

# pip install https://github.com/lancopku/pkuseg-python/archive/master.zip
import pkuseg

# jieba/pkuseg
cfg = {"segmenter": "pkuseg"}
nlp = Chinese.from_config({"nlp": {"tokenizer": cfg}})

# pkuseg_model="spacy_ontonotes" pkuseg_model="news" medicine tourism web mixed
nlp.tokenizer.initialize(pkuseg_model="web")

nlp.tokenizer.pkuseg_update_user_dict(["杭州大厦"])

print(nlp.tokenizer.pkuseg_seg)
#spacy_pkuseg.pkuseg(postag=True)

text = "我骑车去杭州大厦吃饭"
doc = nlp(text)

for token in doc:
    print(token.text, token.dep_, token.pos_, token.tag_)
