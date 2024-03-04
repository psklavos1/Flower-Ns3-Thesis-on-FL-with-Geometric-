# third party
import tensorflow as tf

# built-in
import math
import random
import numpy as np

# In this function we want to partition our data in the number of clients we have
# the function returns dataloaders so we need to return specific batch size.
# The val ratio is the ration of samples to be put aside for validation check
# we get one dataloader for each client to use either for train or validation
def prepare_dataset(
    num_partitions: int, has_trainval_support=False, val_ratio: float = 0.1
):
    """Download and partitions the MNIST dataset."""
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    # Normalize both training and test datasets
    x_train, x_test = x_train / 255.0, x_test / 255.0

    partitions = []
    # We keep all partitions equal-sized in this example
    partition_size = math.floor(len(x_train) / num_partitions)
    for cid in range(num_partitions):
        # Split dataset into non-overlapping NUM_CLIENT partitions
        idx_from, idx_to = cid * partition_size, (cid + 1) * partition_size
        partitions.append((x_train[idx_from:idx_to], y_train[idx_from:idx_to]))
        # Now that data are split we create dataloaders with train val support to return to clients

    if has_trainval_support:
        trainsets, validationsets = trainval_support(partitions, val_ratio)
    else:
        trainsets = partitions
        validationsets = []

    testset = (x_test, y_test)

    return trainsets, validationsets, testset


def trainval_support(partitions, val_ratio):
    trainsets = []
    validationsets = []

    for partition in partitions:
        x_partition, y_partition = partition

        # Calculate the number of samples for train and validation
        num_samples = len(x_partition)
        num_train_samples = int(num_samples * (1 - val_ratio))

        # Shuffle the indices to randomly select train and validation samples
        indices = list(range(num_samples))
        random.shuffle(indices)

        train_indices = indices[:num_train_samples]
        validation_indices = indices[num_train_samples:]

        # Create train and validation data based on the shuffled indices
        x_train_partition = x_partition[train_indices]
        y_train_partition = y_partition[train_indices]
        x_validation_partition = x_partition[validation_indices]
        y_validation_partition = y_partition[validation_indices]

        # Append the train and validation data to their respective lists
        trainsets.append((x_train_partition, y_train_partition))
        validationsets.append((x_validation_partition, y_validation_partition))

    return trainsets, validationsets


# Functions to get a client dataset
def _load_dataset(ds_name):
    """Load dataset based on the given name."""
    datasets = {
        "mnist": tf.keras.datasets.mnist.load_data,
        "fashion_mnist": tf.keras.datasets.fashion_mnist.load_data,
        "cifar10": tf.keras.datasets.cifar10.load_data,
        "cifar100": tf.keras.datasets.cifar100.load_data,
    }

    if ds_name not in datasets:
        raise ValueError(f"Invalid dataset name '{ds_name}'")
    return datasets[ds_name]()


def _preprocess_data(x_train, x_test, ds_name):
    """Preprocess data based on the dataset type."""
    if ds_name in ["mnist", "fashion_mnist"]:
        x_train = x_train.reshape(-1, 28, 28, 1).astype("float32") / 255.0
        x_test = x_test.reshape(-1, 28, 28, 1).astype("float32") / 255.0
    elif ds_name == "cifar10":
        x_train = x_train.astype("float32") / 255.0
        x_test = x_test.astype("float32") / 255.0
    return x_train, x_test


def _shuffle_data(x_train, y_train, seed=20):
    """Shuffle the training data with a fixed seed."""
    indices = tf.range(start=0, limit=tf.shape(x_train)[0], dtype=tf.int32)
    shuffled_indices = tf.random.shuffle(indices, seed=seed)
    return tf.gather(x_train, shuffled_indices), tf.gather(y_train, shuffled_indices)


def _partition_data(x_train, y_train, num_partitions, partition_index):
    """Partition the training data."""
    partition_size = len(x_train) // num_partitions
    start_idx = partition_index * partition_size
    end_idx = (
        start_idx + partition_size
        if partition_index < num_partitions - 1
        else len(x_train)
    )
    return x_train[start_idx:end_idx], y_train[start_idx:end_idx]


# Non-IID data Support
def _split_data_by_class(x, y):
    class_data = {}
    for unique_class in np.unique(y):
        class_indices = np.where(y == unique_class)[0]
        class_data[unique_class] = (x[class_indices], y[class_indices])
    return class_data


import numpy as np


def _partition_data_with_bias(
    class_data, class_percentages, num_partitions, partition_index
):
    x_partition = []
    y_partition = []

    # Calculate the fixed size for each partition
    total_dataset_size = sum(len(x_data) for x_data, _ in class_data.values())
    partition_size = total_dataset_size // num_partitions

    for cls, (x_data, y_data) in class_data.items():
        # Determine the number of samples to include from this class based on its percentage
        num_samples_from_class = int(partition_size * class_percentages[cls])

        # Calculate the starting index for the current partition
        start_index = (partition_size * partition_index) % len(x_data)
        end_index = start_index + num_samples_from_class

        # Handle wrap-around if the end index exceeds the array length
        if end_index <= len(x_data):
            x_selected = x_data[start_index:end_index]
            y_selected = y_data[start_index:end_index]
        else:
            x_selected = np.concatenate(
                (x_data[start_index:], x_data[: end_index % len(x_data)])
            )
            y_selected = np.concatenate(
                (y_data[start_index:], y_data[: end_index % len(y_data)])
            )

        # Add to the partition
        x_partition.extend(x_selected)
        y_partition.extend(y_selected)

    # Check if the partition size is reached, and fill in the gap if necessary
    shortfall = partition_size - len(x_partition)
    if shortfall > 0:
        additional_indices = np.random.choice(
            range(total_dataset_size), shortfall, replace=False
        )
        all_x = np.concatenate([x for x, _ in class_data.values()])
        all_y = np.concatenate([y for _, y in class_data.values()])
        x_additional = all_x[additional_indices]
        y_additional = all_y[additional_indices]

        x_partition.extend(x_additional)
        y_partition.extend(y_additional)

    # Shuffle the partition data
    combined = list(zip(x_partition, y_partition))
    np.random.shuffle(combined)
    x_partition[:], y_partition[:] = zip(*combined)

    return x_partition, y_partition


def _generate_random_class_percentages(num_classes, seed=None):
    if seed is not None:
        np.random.seed(seed)
    percentages = np.random.dirichlet(np.ones(num_classes), size=1)[0]
    return {cls: percentage for cls, percentage in enumerate(percentages)}


def _generate_low_bias_class_percentages(num_classes, seed=None):
    if seed is not None:
        np.random.seed(seed)
    base = np.ones(num_classes) / num_classes
    noise = np.random.normal(0, 0.02, num_classes)  # Small variations
    percentages = np.clip(base + noise, 0, 1)
    percentages /= np.sum(percentages)  # Normalize to sum to 1
    return {cls: percentage for cls, percentage in enumerate(percentages)}


def _generate_medium_bias_class_percentages(num_classes, seed=None):
    if seed is not None:
        np.random.seed(seed)
    percentages = np.random.dirichlet(np.ones(num_classes) * 2, size=1)[
        0
    ]  # Adjust concentration parameter for more variance
    return {cls: percentage for cls, percentage in enumerate(percentages)}


def _generate_high_bias_class_percentages(num_classes, seed=None):
    if seed is not None:
        np.random.seed(seed)
    focus_classes_count = np.random.choice([1, 2, 3], 1)[0]
    focus_classes = np.random.choice(num_classes, focus_classes_count, replace=False)
    percentages = np.zeros(num_classes)

    for focus_class in focus_classes:
        percentages[focus_class] = np.random.uniform(
            0.25, 0.45
        )  # Adjust these bounds as needed

    if np.sum(percentages) > 1:
        percentages = percentages / np.sum(percentages)

    remaining_percentage = 1 - np.sum(percentages)
    remaining_classes_count = num_classes - focus_classes_count

    for i in range(num_classes):
        if i not in focus_classes:
            percentages[i] = remaining_percentage / remaining_classes_count

    return {cls: percentage for cls, percentage in enumerate(percentages)}


def _print_class_percentages(percentages):
    # print("Bias Percentages:")
    # print(f"{percentages}")
    for cls, percentage in percentages.items():
        print(f"  Class {cls}: {percentage*100:.2f}%")


def generate_class_percentages(ds_name, bias_template, seed=None):
    if ds_name == "cifar100":
        num_classes = 100
    else:  # Default to 10 for MNIST and Fashion MNIST and cifar10
        num_classes = 10

    percentages = None
    if bias_template == 0:
        percentages = _generate_random_class_percentages(num_classes, seed=seed)
    elif bias_template == 1:
        percentages = _generate_low_bias_class_percentages(num_classes, seed=seed)
    elif bias_template == 2:
        percentages = _generate_medium_bias_class_percentages(num_classes, seed=seed)
    elif bias_template == 3:
        percentages = _generate_high_bias_class_percentages(num_classes, seed=seed)
    else:
        raise ValueError("Invalid bias_template value")

    _print_class_percentages(percentages)
    return percentages


def get_dataset(
    num_partitions,
    partition_index,
    ds_name="mnist",
    non_iid=False,
    class_percentages=None,
):
    """Load and preprocess dataset based on ds_name, then partition it."""
    (x_train, y_train), (x_test, y_test) = _load_dataset(ds_name)
    x_train, x_test = _preprocess_data(x_train, x_test, ds_name)

    if non_iid:
        assert class_percentages is not None
        # Split the data by class and distribute according to the specified percentages
        class_data = _split_data_by_class(x_train, y_train)
        x_train_partition, y_train_partition = _partition_data_with_bias(
            class_data,
            class_percentages,
            num_partitions,
            partition_index,
        )
    else:
        x_train_shuffled, y_train_shuffled = _shuffle_data(x_train, y_train)
        x_train_partition, y_train_partition = _partition_data(
            x_train_shuffled, y_train_shuffled, num_partitions, partition_index
        )
    return (x_train_partition, y_train_partition), (x_test, y_test)
