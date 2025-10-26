from preprocess.clean_data import Cleaner
from preprocess.vectorization import Vectorizer
import joblib as jb

class Dep_pipeline:
    def run(self, txt):
        cl = Cleaner(txt)
        X = cl.Clean_Resume()
        
        vec = Vectorizer()
        X = vec.load_vectorizer(txt)
        
        
        model = jb.load(r'\models\KNC model.pkl')
        predict = model.predict(X)
        
        lb = jb.load(r'\models\encoding labels.pkl')
        prediction_name = lb.inverse_transform(predict)
        return prediction_name
        
