import spacy
from spacy import displacy

nlp = spacy.load('../output/model-best')

doc = nlp("我要买iPhone X和mac")
for token in doc.ents:
    print(token)

displacy.serve(doc, style="ent", port=9797)
