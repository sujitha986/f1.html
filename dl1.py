import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense
# Generate training data
x_train = np.random.randint(0, 100, (1000, 2)) # Two random integers between 0 and 100
y_train = np.sum(x_train, axis=1) # Sum of the two integers
# Create the model
model = Sequential()
model.add(Dense(units=10, activation='relu', input_dim=2))
model.add(Dense(units=1))
# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])
# Train the model
model.fit(x_train, y_train, epochs=100, batch_size=10)
# Test the model on new data
test_numbers = np.array([[45, 15], [60, 30], [23, 77], [10, 90]]) # Example pairs of numbers
predictions = model.predict(test_numbers)
# Print the results
for i, prediction in enumerate(predictions):
 print(f"{test_numbers[i][0]} + {test_numbers[i][1]} = {prediction[0]}")
