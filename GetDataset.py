from datetime import timedelta
import time
from keras.utils import image_dataset_from_directory
from keras.config import image_data_format

BATCH_SIZE = 64
IMAGE_SIZE = (32, 32)  # Original size is (128, 128)


def print_elapsed_time(message="{0}", start_time: float | None = None):
    if start_time is not None:
        elapsed = time.perf_counter() - start_time
        print(message.format(timedelta(seconds=elapsed)))
    return time.perf_counter()


def print_dataset_shape(dataset):
    print(f"- Length: {len(dataset)}, Type: {type(dataset)}")
    for image_batch, labels_batch in dataset:
        print(f"+ Shape: {image_batch.shape}, Type: {type(image_batch)}")
        print(f"+ Shape: {labels_batch.shape}, Type: {type(labels_batch)}")
        break


def get_dataset(directory: str):
    start_time = print_elapsed_time()
    training_ds, validation_ds = image_dataset_from_directory(
        directory,
        label_mode="categorical",  # categorical_crossentropy (one-hot array)
        color_mode="grayscale",
        batch_size=BATCH_SIZE,  # Default is 32
        image_size=IMAGE_SIZE,
        seed=42,
        validation_split=0.2,
        subset="both",
        crop_to_aspect_ratio=True,
        verbose=True,
    )  # About 2 minutes
    print_elapsed_time("- Finished loading dataset in {0}", start_time)

    print_dataset_shape(training_ds)
    print_dataset_shape(validation_ds)

    return training_ds, validation_ds
