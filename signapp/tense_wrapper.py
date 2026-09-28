class TenseModelWrapper:
    def __init__(self, model, vocab, label_encoder):
        self.model = model
        self.vocab = vocab
        self.label_encoder = label_encoder
        self.model.eval()

    def preprocess(self, sentence):
        import re
        sentence = re.sub(r'[^\u0980-\u09FF\s]', '', sentence)
        sentence = re.sub(r'\s+', ' ', sentence).strip()

        unk_id = self.vocab.get('<UNK>', 0)
        tokens = [self.vocab.get(w, unk_id) for w in sentence.split()]

        # empty sentence হলে crash এড়াতে
        if not tokens:
            tokens = [0]

        return tokens

    def predict(self, texts):
        import torch
        results = []

        for sentence in texts:
            tokens = self.preprocess(sentence)
            tensor = torch.tensor(tokens, dtype=torch.long).unsqueeze(0)

            with torch.no_grad():
                output = self.model(tensor)
                pred = torch.argmax(output, dim=1).item()

            # index কে original label-এ convert করো
            label = self.label_encoder.inverse_transform([pred])[0]
            results.append(label)

        return results