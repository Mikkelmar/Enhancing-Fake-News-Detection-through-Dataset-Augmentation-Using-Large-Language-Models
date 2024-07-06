import tensorflow as tf

def undersample_majority_class(dataset):
    # Split dataset into positive and negative examples
    positive_samples = dataset.filter(lambda x, y: tf.equal(y, 1))
    negative_samples = dataset.filter(lambda x, y: tf.equal(y, 0))

    # Count the number of positive and negative samples
    positive_count = positive_samples.reduce(0, lambda x, _: x + 1).numpy()
    negative_count = negative_samples.reduce(0, lambda x, _: x + 1).numpy()

    # Debug prints
    print(f"Number of positive samples: {positive_count}")
    print(f"Number of negative samples: {negative_count}")

    # Determine the number of samples to take from each class
    min_count = min(positive_count, negative_count)
    print(f"Number of samples to take from each class: {min_count}")

    # Shuffle and take min_count samples from each class
    positive_samples = positive_samples.shuffle(positive_count, seed=42).take(min_count)
    negative_samples = negative_samples.shuffle(negative_count, seed=42).take(min_count)

    # Debug prints to check the counts after shuffling and taking samples
    print(f"Number of positive samples after undersampling: {positive_samples.reduce(0, lambda x, _: x + 1).numpy()}")
    print(f"Number of negative samples after undersampling: {negative_samples.reduce(0, lambda x, _: x + 1).numpy()}")

    # Combine the undersampled datasets
    balanced_dataset = positive_samples.concatenate(negative_samples)

    # Shuffle the balanced dataset
    balanced_dataset = balanced_dataset.shuffle(min_count * 2, seed=42)
    
    # Debug print to check the final size of the balanced dataset
    final_count = balanced_dataset.reduce(0, lambda x, _: x + 1).numpy()
    print(f"Total number of samples in balanced dataset: {final_count}")

    return balanced_dataset, min_count * 2