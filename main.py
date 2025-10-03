# importojm librarit si pandas skicit learn

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

from sklearn.metrics import classification_report

# thirrim filen me pandas
df = pd.read_csv("SMSSpamCollection", sep="\t", header=None, names=["label", "text"])


# Printimi
print("Numri i lajmeve", len(df))
print("\nNumri i lajmeve sipas kategoris:\n")
print(df["label"].value_counts())



# Me tf dhe idf ktyhejm tekstet ne numra
vic = TfidfVectorizer(lowercase=True,max_features=2000)
X =  vic.fit_transform(df["text"])
y = df["label"]

print("Madhesia e matixes eshte: ",X.shape)

#Ndarje e datasetit ne train edhe test
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=99)
print("Train: ",X_train.shape[0])
print("Test: ", X_test.shape[0])

# trajnimi i modelit me multinomialNB
model = MultinomialNB()
model.fit(X_train, y_train)

#Pritimi i saktesis se modelit
y_pred = model.predict(X_test)
print("Saktësia e modelit:", accuracy_score(y_test, y_pred))

#Raporti i plot
print("\nRaport i plotë:\n", classification_report(y_test, y_pred))



def predict_message(message):
    vec = vic.transform([message])
    return model.predict(vec)[0]



while True:
    msg = input("\nShkruaj një mesazh (ose 'q' per me dale): ")
    if msg.lower() == "q":
        break
    print("Parashikimi:", predict_message(msg))
