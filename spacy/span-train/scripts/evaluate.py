import spacy

nlp = spacy.load('../output/model-best')

doc = nlp("我要买iPhone X")
for token in doc.ents:
    print(token)
