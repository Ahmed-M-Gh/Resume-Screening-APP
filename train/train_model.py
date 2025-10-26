from preprocess.clean_data import Cleaner
from preprocess.vectorization import Vectorizer
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import accuracy_score
import joblib as jb
df = pd.read_csv(r'data/ResumeDataSet.csv')

print('[Cleaning] start cleaning....')
df['Resume'] = df['Resume'].apply(lambda x: Cleaner(x).Clean_Resume())
print('[Cleaning] cleaning completed successfully')

print('[Encoding] start encoding target column....')
lb = LabelEncoder()
lb.fit(df['Category'])
Y = lb.transform(df['Category'])
jb.dump(lb, r'models/encoding labels.pkl')
print('[Encoding] encoding target column completed successfully')

print('[Vectorization] start to vectorization the resume....')
vec = Vectorizer()
X = vec.fit_on_df(df)
print('[Vectorization] vectorization completed successfully')

print('[Training] start training model....')
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.33, random_state=42)

KNC = OneVsRestClassifier(KNeighborsClassifier())
KNC.fit(x_train, y_train)
y_pred = KNC.predict(x_test)
print('[Training] training completed successfully')
acc = accuracy_score(y_test, y_pred)
print(f'[Evaluation] accuracy score = {acc}')


print('[Saving] saving model....')
jb.dump(KNC, r'models/KNC model.pkl')
print('[Saving] model saved successfully')
