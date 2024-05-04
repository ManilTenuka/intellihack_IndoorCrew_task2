import json

def load_intent_data(filepath):
    with open(filepath, 'r') as file:
        data = json.load(file)
    texts = []
    labels = []
    for intent, examples in data['intents'].items():
        for example in examples:
            texts.append(example)
            labels.append(intent)
    return texts, labels
