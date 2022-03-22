## 模型训练

1.初始化生成配置文件

```python
python - m
spacy
init
config. / config / config.cfg - -lang
zh - -pipeline
ner
```

2. 运行span_train.py,生成训练数据
3. 训练模型

```python
python - m
spacy
train. / config / config.cfg - -output. / output - -paths.train. / data / train.spacy - -paths.dev. / data / dev.spacy
```

4. 调用evaluate.py，测试训练好的模型