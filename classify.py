import sys
from utils.model_utils import load_model

def classify_text(model, text):
    probs = model.predict_proba([text])[0]
    max_prob = max(probs)
    predicted_intent = model.classes_[probs.argmax()]
    return predicted_intent, max_prob

if __name__ == "__main__":
    model = load_model('models/intent_classifier.pkl')
    text = sys.argv[1] if len(sys.argv) > 1 else "Hello"
    intent, confidence = classify_text(model, text)
    threshold = 0.7
    if confidence < threshold:
        print("NLU fallback: Intent could not be confidently determined. confidence Level: ", confidence)
    else:
        print(f"Intent: {intent}, Confidence: {confidence:.2f}")
