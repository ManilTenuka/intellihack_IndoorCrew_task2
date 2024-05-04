from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from utils.data_utils import load_intent_data
from utils.model_utils import save_model

def train_model(data_path, model_path):
    texts, labels = load_intent_data(data_path)
    model = make_pipeline(CountVectorizer(), LogisticRegression(max_iter=1000))
    model.fit(texts, labels)
    save_model(model, model_path)

if __name__ == "__main__":
    train_model('data/intent_data.json', 'models/intent_classifier.pkl')