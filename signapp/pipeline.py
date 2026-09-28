# In signapp/pipeline.py (or wherever it's supposed to be)


def get_pipe():
   # Your model pipeline definition goes here
   from sklearn.pipeline import Pipeline
   from sklearn.feature_extraction.text import CountVectorizer
   from sklearn.linear_model import LogisticRegression


   # Example pipeline
   pipe = Pipeline([
       ('vectorizer', CountVectorizer()),
       ('classifier', LogisticRegression())
   ])


   return pipe
