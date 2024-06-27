import lipnn

dependencies = ['torch']

models = filter(lambda name: name.startswith("cifar"), dir(lipnn))
globals().update({model: getattr(lipnn, model) for model in models})