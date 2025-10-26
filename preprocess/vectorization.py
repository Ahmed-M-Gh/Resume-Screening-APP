from sklearn.feature_extraction.text import TfidfVectorizer
import joblib as jb

class Vectorizer:
    def __init__(self):
        self.tfidf = TfidfVectorizer(stop_words='english')
        
        
    def fit_on_df(self, df):
        self.tfidf.fit(df['Resume'])
        Required = self.tfidf.transform(df['Resume'])
        jb.dump(self.tfidf, r'\models\vectorizer.pkl')
        return Required
    
    def load_vectorizer(self, txt):
        self.tfidf = jb.load(r'\models\vectorizer.pkl')
        txt = self.tfidf.transform([txt])
        txt = txt.toarray()
        return txt
