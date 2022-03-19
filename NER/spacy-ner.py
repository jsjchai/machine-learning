from __future__ import unicode_literals, print_function

import random
from pathlib import Path
import spacy
from spacy.util import minibatch, compounding
import pickle
from spacy.training.example import Example
from tqdm import tqdm

TRAIN_DATA = [
    ("两肺多发转移", {"entities": [(0, 2, "肺")]}),
    ("转移左肺", {"entities": [(2, 4, "肺")]})
]

model = None
output_dir = Path("ner/")
n_iter = 100

# load the model

if model is not None:
    nlp = spacy.load(output_dir)
    print("Loaded model '%s'" % model)
else:
    nlp = spacy.blank('zh')
    print("Created blank 'zh' model")

ner = None
if 'ner' not in nlp.pipe_names:
    ner = nlp.add_pipe('ner')
else:
    ner = nlp.get_pipe('ner')

for _, annotations in TRAIN_DATA:
    for ent in annotations.get('entities'):
        ner.add_label(ent[2])

example = []
other_pipes = [pipe for pipe in nlp.pipe_names if pipe != 'ner']
sizes = compounding(1.0, 4.0, 1.001)
with nlp.disable_pipes(*other_pipes):  # only train NER
    optimizer = nlp.begin_training()
    for itn in range(n_iter):
        random.shuffle(TRAIN_DATA)
        losses = {}
        for text, annotations in tqdm(TRAIN_DATA):
            doc = nlp.make_doc(text)
            example = Example.from_dict(doc, annotations)
            nlp.update([example], sgd=optimizer, drop=0.35, losses=losses)
            print("losses:", losses)

if output_dir is not None:
    output_dir = Path(output_dir)
    if not output_dir.exists():
        output_dir.mkdir()
    nlp.to_disk(output_dir)
    print("Saved model to", output_dir)
pickle.dump(nlp, open("education nlp.pkl", "wb"))

doc = nlp("全肺未见转移")
for ent in doc.ents:
    print(ent.label_ + '  ------>   ' + ent.text)
