import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt 
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.model_selection import train_test_split

insurance = pd.read_csv('https://raw.githubusercontent.com/stedy/Machine-Learning-with-R-datasets/master/insurance.csv')

ct = make_column_transformer(
    (MinMaxScaler(),['age', 'bmi','children']), 
    (OneHotEncoder(handle_unknown='ignore'),['sex','smoker','region'])
)

X = insurance.drop("charges", axis=1)
y = insurance["charges"]

x_train, x_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=42)

ct.fit(x_train)

x_train_normal = ct.transform(x_train)
x_test_normal = ct.transform(x_test)

#set random state
tf.random.set_seed(42)

# 1 create a model 
model = tf.keras.Sequential([
    tf.keras.layers.Dense(100),
    tf.keras.layers.Dense(10),
    tf.keras.layers.Dense(1)
])

# 2 compile the model
model.compile(loss=tf.losses.mae,
              optimizer=tf.optimizers.Adam(lr=0.01),
              metrics=['mae'])

# 3 fit the model
model.fit(x_train_normal,y_train,epochs=100)

#evaluate the model
model.evaluate(x_test_normal, y_test)