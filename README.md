# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the packages.
2. Analyse the data.
3. Use modelselection and Countvectorizer to preditct the values.
4. Find the accuracy and display the result.

## Program:
```
/*
Program to implement the SVM For Spam Mail Detection..
Developed by: Aman Singh
RegisterNumber: 212224040020
*/
```
```python
import chardet
file = "/content/spam.csv"
with open(file, 'rb') as rawdata:
    result = chardet.detect(rawdata.read(100000))
print(result)

import pandas as pd
data = pd.read_csv( "/content/spam.csv", encoding='windows-1252')
print(data.head())
print(data.info())
print(data.isnull().sum())

x = data["v2"].values 
y = data["v1"].values

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()
x_train = cv.fit_transform(x_train)
x_test = cv.transform(x_test)


from sklearn.svm import SVC
svc = SVC()
svc.fit(x_train, y_train)
y_pred = svc.predict(x_test)
print(y_pred)

from sklearn import metrics
accuracy = metrics.accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
```

## Output:
<img width="1116" alt="Screenshot 2025-05-13 at 2 27 44 PM" src="https://github.com/user-attachments/assets/6c8fb29c-ccc6-4765-b02f-e2181f7088d0" />
<br>

<img width="611" alt="Screenshot 2025-05-13 at 2 27 54 PM" src="https://github.com/user-attachments/assets/4bdb607d-89ad-443e-99fb-09af09ccc913" />
<br>
<img width="517" alt="Screenshot 2025-05-13 at 2 28 06 PM" src="https://github.com/user-attachments/assets/e6bc2da9-4e7a-4e8d-baf1-115376c61ccd" />
<br>

<img width="374" alt="Screenshot 2025-05-13 at 2 28 28 PM" src="https://github.com/user-attachments/assets/5bcbffde-3afe-4f27-995b-f9604c8723a5" />
<br>
<img width="782" alt="Screenshot 2025-05-13 at 2 29 05 PM" src="https://github.com/user-attachments/assets/871be893-dd19-4c92-8bbd-701ddedbec96" />
<br>
<img width="340" alt="Screenshot 2025-05-13 at 2 29 20 PM" src="https://github.com/user-attachments/assets/b8bba358-f109-426d-bc48-fe9f4a4c9371" />
<br>

## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
