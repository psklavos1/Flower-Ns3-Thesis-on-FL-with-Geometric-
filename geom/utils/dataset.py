# third party
import tensorflow as tf
import keras

# built-in
import math
import random
import numpy as np

# ==============================================================================================
# * Main Functions for dataset loading to be used publicaly


def generate_class_percentages(ds_name, bias_template, seed=None):
    """
    This function creates a distribution of class percentages for a given dataset,
    allowing for the simulation of different bias levels in the data distribution.
    It supports generating random, low, medium, and high bias distributions.

    Parameters:
        ds_name (str): The name of the dataset. Currently supports 'cifar100', 'MNIST',
            'Fashion MNIST', and 'cifar10'.
        bias_template (int): Specifies the bias level in the class distribution.
            - 0: Random bias
            - 1: Low bias
            - 2: Medium bias
            - 3: High bias
            Raises ValueError for invalid bias_template values.
        seed (int, optional): The seed for random number generation to ensure
            reproducibility. Defaults to None.

    Returns:
        list: A list of percentages representing the class distribution in the dataset.
            The list length corresponds to the number of classes in the dataset.

    Raises:
        ValueError: If an invalid `bias_template` value is passed.
    """
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
    validation_split=0.1,
    train_val_split=0.0,
    seed=20,
):
    """
    Load, preprocess, and partition a dataset for federated learning scenarios.

    This function handles the loading and preprocessing of a specified dataset
    (e.g., MNIST, Fashion MNIST, cifar10, cifar100). It supports partitioning the data into
    multiple parts for simulation of distributed data sources in federated learning.
    It also provides the option to create partitions with non-IID data distributions
    based on provided class percentages, and can optionally create a validation set.

    Parameters:
        num_partitions (int): The number of partitions to split the dataset into.
        partition_index (int): The index of the partition to return. This is used
            to simulate distributing different parts of the dataset to different clients.
        ds_name (str, optional): The name of the dataset to load. Defaults to 'mnist'.
        non_iid (bool, optional): Whether to partition the data in a non-IID manner,
            based on class percentages. Defaults to False.
        class_percentages (list, optional): The percentages for each class to be included
            in each partition, used only when non_iid is True. Defaults to None.
        validation (bool, optional): Whether to split off a portion of the training data
            into a validation set. Defaults to False.
        validation_split (float, optional): The fraction of the training data to be used
            as validation data, if validation is True. Defaults to 0.1.

    Returns:
        tuple: A tuple containing the training, testing, and (optionally) validation
            tf.data.Dataset objects. If validation is False, the validation dataset
            in the tuple will be None.

    Raises:
        AssertionError: If non_iid is True but class_percentages is None.
    """

    train_dataset: tf.data.Dataset
    validation_dataset: tf.data.Dataset
    train_val_dataset: tf.data.Dataset
    (x_train, y_train), _ = _load_and_preprocess_dataset(ds_name)

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
        # needed to create a ranodom dataset.
        x_train_partition, y_train_partition = _shuffle_and_partition_data(
            x_train, y_train, num_partitions, partition_index, seed
        )

    # Validation set for client coming from his local trainset distribution.

    (
        x_train_partition,
        y_train_partition,
        x_val_partition,
        y_val_partition,
        x_train_val_partition,
        y_train_val_partition,
    ) = _dataset_split(
        x_train_partition, y_train_partition, validation_split, train_val_split
    )

    # Convert to tf.data.Dataset for training
    train_dataset = _prep_dataset(x_train_partition, y_train_partition, repeat=True)
    validation_dataset = _prep_dataset(x_val_partition, y_val_partition)
    train_val_dataset = _prep_dataset(x_train_val_partition, y_train_val_partition)

    return train_dataset, validation_dataset, train_val_dataset


def _prep_dataset(x_partition, y_partition, repeat=False) -> tf.data.Dataset:
    dataset = None
    if len(x_partition) != 0:
        dataset = tf.data.Dataset.from_tensor_slices((x_partition, y_partition))
        if repeat:
            dataset = dataset.cache().shuffle(buffer_size=len(x_partition)).repeat()
        else:
            dataset = dataset.cache().shuffle(buffer_size=len(x_partition))

    return dataset


def get_testset(ds_name="mnist"):
    """
    Load and preprocess the test set for the specified dataset.

    This function loads the test set of the specified dataset and preprocesses it
    to be ready for evaluation.

    Args:
        ds_name (str, optional): The name of the dataset to load. Defaults to 'mnist'.

    Returns:
        tf.data.Dataset: The preprocessed test dataset ready for evaluation.
    """
    _, (x_test, y_test) = _load_and_preprocess_dataset(ds_name)

    # Convert to tf.data.Dataset for evaluation
    test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))
    test_dataset = test_dataset.cache()

    return test_dataset


# ==============================================================================================
# * Utility Functions


def _load_and_preprocess_dataset(ds_name):
    """
    Load and preprocess a specified dataset.

    This function loads a dataset by its name from the available TensorFlow datasets and
    applies the necessary preprocessing steps to prepare the data for training and testing.

    Args:
        ds_name (str): Name of the dataset to load. Supported values are 'mnist',
                       'fashion_mnist', 'cifar10', 'cifar100'.

    Returns:
        tuple: A tuple containing preprocessed training data (x_train, y_train) and
               testing data (x_test, y_test).

    Raises:
        ValueError: If `ds_name` is not a recognized dataset name.
    """
    datasets = {
        "mnist": keras.datasets.mnist.load_data,
        "fashion_mnist": keras.datasets.fashion_mnist.load_data,
        "cifar10": keras.datasets.cifar10.load_data,
        "cifar100": keras.datasets.cifar100.load_data,
    }

    if ds_name not in datasets:
        raise ValueError(f"Invalid dataset name '{ds_name}'")

    (x_train, y_train), (x_test, y_test) = datasets[ds_name]()

    if ds_name in ["mnist", "fashion_mnist"]:
        x_train = x_train.reshape(-1, 28, 28, 1).astype("float32") / 255.0
        x_test = x_test.reshape(-1, 28, 28, 1).astype("float32") / 255.0
    elif ds_name in ["cifar10", "cifar100"]:
        x_train = x_train.astype("float32") / 255.0
        x_test = x_test.astype("float32") / 255.0

    return (x_train, y_train), (x_test, y_test)


def _shuffle_and_partition_data(
    x_train, y_train, num_partitions, partition_index, seed=20
):
    """
    Shuffle and partition the training data.

    This function shuffles the training data and partitions it into a specified number of
    partitions. It returns the partition corresponding to the given partition index.

    Args:
        x_train (ndarray): Training features.
        y_train (ndarray): Training labels.
        num_partitions (int): Number of partitions to split the data into.
        partition_index (int): Index of the partition to return.
        seed (int, optional): Seed for reproducibility of the shuffling. Defaults to 20.

    Returns:
        tuple: A tuple containing the features (x_train_partition) and labels (y_train_partition)
               for the specified partition, as numpy arrays.
    """
    dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    dataset = dataset.shuffle(buffer_size=len(x_train), seed=seed).batch(
        len(x_train) // num_partitions
    )

    partitioned_data = list(dataset)
    x_train_partition, y_train_partition = partitioned_data[partition_index]

    return x_train_partition.numpy(), y_train_partition.numpy()


# * For non-iid Support
def _split_data_by_class(x, y):
    """
    Split dataset by class.

    Args:
        x (ndarray): Features.
        y (ndarray): Labels.

    Returns:
        Dictionary mapping each class to its corresponding features and labels.
    """
    class_data = {}
    for unique_class in np.unique(y):
        class_indices = np.where(y == unique_class)[0]
        class_data[unique_class] = (x[class_indices], y[class_indices])
    return class_data


def _partition_data_with_bias(
    class_data, class_percentages, num_partitions, partition_index
):
    """
    Partition data with specified class bias.

    Args:
        class_data (dict): Data split by class.
        class_percentages (dict): Target percentages for each class.
        num_partitions (int): Number of partitions.
        partition_index (int): Index of the current partition.

    Returns:
        Tuple of partitioned features and labels with the specified bias.
    """
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
    """
    Generate random class percentages for non-IID data simulation.

    Args:
        num_classes (int): Number of classes.
        seed (int, optional): Seed for reproducibility.

    Returns:
        Dictionary with class indices as keys and percentages as values.
    """
    if seed is not None:
        np.random.seed(seed)
    percentages = np.random.dirichlet(np.ones(num_classes), size=1)[0]
    return {cls: percentage for cls, percentage in enumerate(percentages)}


def _generate_low_bias_class_percentages(num_classes, seed=None):
    """
    Generate low bias class percentages for non-IID data simulation.

    Args:
        num_classes (int): Number of classes.
        seed (int, optional): Seed for reproducibility.

    Returns:
        Dictionary with class indices as keys and percentages as values.
    """
    if seed is not None:
        np.random.seed(seed)
    base = np.ones(num_classes) / num_classes
    noise = np.random.normal(0, 0.02, num_classes)  # Small variations
    percentages = np.clip(base + noise, 0, 1)
    percentages /= np.sum(percentages)  # Normalize to sum to 1
    return {cls: percentage for cls, percentage in enumerate(percentages)}


def _generate_medium_bias_class_percentages(num_classes, seed=None):
    """
    Generate medium bias class percentages for non-IID data simulation.

    Args:
        num_classes (int): Number of classes.
        seed (int, optional): Seed for reproducibility.

    Returns:
        Dictionary with class indices as keys and percentages as values.
    """
    if seed is not None:
        np.random.seed(seed)
    percentages = np.random.dirichlet(np.ones(num_classes) * 2, size=1)[
        0
    ]  # Adjust concentration parameter for more variance
    return {cls: percentage for cls, percentage in enumerate(percentages)}


def _generate_high_bias_class_percentages(num_classes, seed=None):
    """
    Generate high bias class percentages for non-IID data simulation.

    Args:
        num_classes (int): Number of classes.
        seed (int, optional): Seed for reproducibility.

    Returns:
        Dictionary with class indices as keys and percentages as values.
    """
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


def _dataset_split(x_train, y_train, validation_split=0.1, train_val_split=0.0):
    """
    Split training data into training, validation, and train_val sets.

    Args:
        x_train (ndarray): Original training features.
        y_train (ndarray): Original training labels.
        validation_split (float): Proportion of the data to use for validation.
        train_val_split (float): Proportion of the data to use for train_val.

    Returns:
        Tuples of new training features, new training labels, validation features, validation labels,
        train_val features, and train_val labels.
    """
    # Ensure x_train and y_train are numpy arrays
    x_train = np.array(x_train)
    y_train = np.array(y_train)

    # Shuffle the training data
    shuffled_indices = np.random.permutation(len(x_train)).astype(int)
    x_train_shuffled = x_train[shuffled_indices]
    y_train_shuffled = y_train[shuffled_indices]

    # Calculate the number of samples for validation and train_val
    num_validation_samples = int(len(x_train) * validation_split)
    num_train_val_samples = int(len(x_train) * train_val_split)

    # Split the data
    x_val = x_train_shuffled[:num_validation_samples]
    y_val = y_train_shuffled[:num_validation_samples]
    x_train_val = x_train_shuffled[
        num_validation_samples : num_validation_samples + num_train_val_samples
    ]
    y_train_val = y_train_shuffled[
        num_validation_samples : num_validation_samples + num_train_val_samples
    ]
    x_train_new = x_train_shuffled[num_validation_samples + num_train_val_samples :]
    y_train_new = y_train_shuffled[num_validation_samples + num_train_val_samples :]

    return x_train_new, y_train_new, x_val, y_val, x_train_val, y_train_val


# * Printing
def _print_class_percentages(percentages):
    """
    Print class percentages to the console.

    Args:
        percentages (dict): Dictionary with class indices as keys and percentages as values.
    """
    # print("Bias Percentages:")
    # print(f"{percentages}")
    for cls, percentage in percentages.items():
        print(f"  Class {cls}: {percentage*100:.2f}%")


# ==============================================================================================
# * Deprecated functions best used with run_simulation()
def prepare_dataset(
    num_partitions: int, has_trainval_support=False, val_ratio: float = 0.1
):
    """
    Prepare and partition the MNIST dataset.

    Partitions the MNIST dataset into a specified number of partitions. Optionally provides support for
    splitting each partition into training and validation sets.

    Args:
        num_partitions (int): The number of partitions to split the dataset into.
        has_trainval_support (bool): If True, split each partition into training and validation sets.
        val_ratio (float): Ratio of the validation set to the total dataset size if validation is enabled.

    Returns:
        tuple: A tuple containing lists of training sets, validation sets (empty if validation is not enabled), and the test set.
    """
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

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
    """
    Split each dataset partition into training and validation sets based on the specified ratio.

    Args:
        partitions (list): List of dataset partitions, where each partition is a tuple (x_partition, y_partition).
        val_ratio (float): Ratio of the validation set to the total dataset size.

    Returns:
        tuple: Two lists of tuples, the first for training sets and the second for validation sets.
    """
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
