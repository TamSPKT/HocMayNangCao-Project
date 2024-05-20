from datetime import datetime
import os
import keras
from keras.models import Sequential
from keras.applications import EfficientNetV2B0
from keras import losses, optimizers, metrics, callbacks, saving
import tensorflow as tf

from GetDataset import get_dataset
from HistoryHelpers import save_history


# import tensorflow_decision_forests as tfdf
# tensorflow_decision_forests\tensorflow\ops\inference\inference.so not found
# https://www.tensorflow.org/decision_forests/known_issues#windows_pip_package_is_not_available

# Check the version of Keras, TensorFlow
print(f"- Version: Keras {keras.__version__}, TensorFlow {tf.__version__}")

IMAGE_SHAPE = (32, 32, 1)  # (128, 128, 1)
LABELS_SIZE = 36
MODEL_NAME = "EfficientNet"
DATASET_PATH = "./dataset"
EPOCHS = 16

# Create a EfficientNet model (5:42 212ms/step)
model = EfficientNetV2B0(
    include_top=True,
    weights=None,  # type: ignore
    input_shape=IMAGE_SHAPE,  # NOTE: Still works with include_top=True
    classes=LABELS_SIZE,
    classifier_activation="softmax",
)
# model.summary()
# NOTE: Nested model will cause load_model error
# base_model = EfficientNetV2B0(
#     include_top=False,
#     weights=None,  # type: ignore
#     input_shape=IMAGE_SHAPE,  # (128, 128, 1)
# )
# model = Sequential()
# model.add(base_model)
# model.add(GlobalAveragePooling2D())
# model.add(Dense(LABELS_SIZE, activation="softmax"))

# Load weights
WEIGHTS_EPOCH = 6
if os.path.exists(f"checkpoint/EfficientNet-{WEIGHTS_EPOCH:03d}.weights.h5"):
    saving.load_weights(
        model, filepath=f"checkpoint/EfficientNet-{WEIGHTS_EPOCH:03d}.weights.h5"
    )
    print(f"- Loaded Model({MODEL_NAME}) weights from Epoch({WEIGHTS_EPOCH:03d})")


# Compile the model
# model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
model.compile(
    loss=losses.CategoricalCrossentropy(label_smoothing=0.1),
    # loss=losses.CategoricalCrossentropy(),
    optimizer="adam",
    metrics=["accuracy"],
)
# model.summary()

# Create a ModelCheckpoint callback
checkpoint = callbacks.ModelCheckpoint(
    filepath="checkpoint/EfficientNet-{epoch:03d}.weights.h5",
    save_weights_only=True,
    monitor="val_accuracy",
    mode="max",
    save_best_only=True,
)

training_ds, validation_ds = get_dataset(DATASET_PATH)  # About 2 minutes

# Train the model
history: callbacks.History = model.fit(
    training_ds, validation_data=validation_ds, epochs=EPOCHS, callbacks=[checkpoint]
)
# Epoch 1/2
#    29/21158 ━━━━━━━━━━━━━━━━━━━━ 9:49:40 2s/step - accuracy: 0.1633 - loss: 3.2489

dt = datetime.now().strftime("%Y%m%d%H%M%S")
saving.save_model(
    model,
    filepath=f"checkpoint/model-{MODEL_NAME}-{dt}.keras",
    # overwrite=False,  # ask the user via an interactive prompt.
)
save_history(history, timestamp=f"{MODEL_NAME}-{dt}")
print("Done!")

# EfficientNet
# Epoch 1/10
# 4363/4363 ━━━━━━━━━━━━━━━━━━━━ 1443s 305ms/step - accuracy: 0.5061 - loss: 2.4325 - val_accuracy: 0.7957 - val_loss: 1.2104
# Epoch 2/10
# 4363/4363 ━━━━━━━━━━━━━━━━━━━━ 1356s 311ms/step - accuracy: 0.7947 - loss: 1.2745 - val_accuracy: 0.8333 - val_loss: 1.0877
# Epoch 3/10
# 4363/4363 ━━━━━━━━━━━━━━━━━━━━ 1467s 336ms/step - accuracy: 0.8346 - loss: 1.0965 - val_accuracy: 0.8660 - val_loss: 0.9963
# Epoch 4/10
# 4363/4363 ━━━━━━━━━━━━━━━━━━━━ 1310s 300ms/step - accuracy: 0.8526 - loss: 1.0420 - val_accuracy: 0.8011 - val_loss: 1.3164
# Epoch 5/10
# 4363/4363 ━━━━━━━━━━━━━━━━━━━━ 1326s 304ms/step - accuracy: 0.8378 - loss: 1.0813 - val_accuracy: 0.8762 - val_loss: 0.9687
# Epoch 6/10
# 4363/4363 ━━━━━━━━━━━━━━━━━━━━ 1320s 302ms/step - accuracy: 0.8640 - loss: 1.0090 - val_accuracy: 0.8782 - val_loss: 0.9967
# Epoch 7/10
# 4363/4363 ━━━━━━━━━━━━━━━━━━━━ 1316s 301ms/step - accuracy: 0.8486 - loss: 1.0522 - val_accuracy: 0.8253 - val_loss: 1.1192
# Epoch 8/10
# 4363/4363 ━━━━━━━━━━━━━━━━━━━━ 1309s 300ms/step - accuracy: 0.8529 - loss: 1.0394 - val_accuracy: 0.8633 - val_loss: 1.0156
# Epoch 9/10
# 4363/4363 ━━━━━━━━━━━━━━━━━━━━ 1313s 301ms/step - accuracy: 0.8584 - loss: 1.0222 - val_accuracy: 0.8653 - val_loss: 1.0588
# Epoch 10/10
# 4363/4363 ━━━━━━━━━━━━━━━━━━━━ 1287s 295ms/step - accuracy: 0.8622 - loss: 1.0141 - val_accuracy: 0.8534 - val_loss: 1.0545
# - History params: {'verbose': 'auto', 'epochs': 10, 'steps': 4363}
# Epoch 7/10
# 4363/4363 ━━━━━━━━━━━━━━━━━━━━ 1417s 292ms/step - accuracy: 0.8580 - loss: 1.0261 - val_accuracy: 0.8649 - val_loss: 1.0033
# Epoch 8/10
# 4363/4363 ━━━━━━━━━━━━━━━━━━━━ 1456s 333ms/step - accuracy: 0.8337 - loss: 1.0974 - val_accuracy: 0.8717 - val_loss: 0.9857
# Epoch 9/10
# 4363/4363 ━━━━━━━━━━━━━━━━━━━━ 1341s 307ms/step - accuracy: 0.8536 - loss: 1.0355 - val_accuracy: 0.8738 - val_loss: 0.9848
# Epoch 10/10
# 4363/4363 ━━━━━━━━━━━━━━━━━━━━ 1348s 309ms/step - accuracy: 0.8567 - loss: 1.0257 - val_accuracy: 0.8753 - val_loss: 0.9829
# Epoch 11/10
# 4363/4363 ━━━━━━━━━━━━━━━━━━━━ 1348s 309ms/step - accuracy: 0.8742 - loss: 0.9789 - val_accuracy: 0.8776 - val_loss: 0.9874
# Epoch 12/10
# 4363/4363 ━━━━━━━━━━━━━━━━━━━━ 1346s 308ms/step - accuracy: 0.8700 - loss: 0.9903 - val_accuracy: 0.8768 - val_loss: 1.0041
# Epoch 13/10
# 4363/4363 ━━━━━━━━━━━━━━━━━━━━ 1345s 308ms/step - accuracy: 0.8544 - loss: 1.0366 - val_accuracy: 0.8554 - val_loss: 1.0527
# Epoch 14/10
# 4363/4363 ━━━━━━━━━━━━━━━━━━━━ 1335s 306ms/step - accuracy: 0.8571 - loss: 1.0246 - val_accuracy: 0.8733 - val_loss: 0.9969
# Epoch 15/10
# 4363/4363 ━━━━━━━━━━━━━━━━━━━━ 1273s 291ms/step - accuracy: 0.8668 - loss: 0.9981 - val_accuracy: 0.8656 - val_loss: 1.0674
# Epoch 16/10
# 4363/4363 ━━━━━━━━━━━━━━━━━━━━ 1290s 295ms/step - accuracy: 0.8726 - loss: 0.9821 - val_accuracy: 0.8763 - val_loss: 1.0941
# - History params: {'verbose': 'auto', 'epochs': 10, 'steps': 4363}
# + Saved History: logs\history-20240505173544.csv
# Done!
