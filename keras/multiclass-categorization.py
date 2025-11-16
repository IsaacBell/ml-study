import Sequential, Layer, Dense from keras

model = keras.Sequential(
  [
    Layer(Dense(64, activation="relu")),
    Layer(Dense(64, activation="relu")),
    Layer(Dense(16, activation="softmax")),
  ]
)

hist = model.compile(
    optimizer="adam",
    category="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

res = model.fit(
    x_train,
    y_train,
    epochs=20,
    batch_size=128,
    validation_data=(x_test, y_test)
)
