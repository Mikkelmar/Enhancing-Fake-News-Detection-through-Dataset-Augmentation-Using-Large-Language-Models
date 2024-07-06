from sklearn.model_selection import train_test_split
from RecreatedFakeStackModel import get_model_with_bert
from keras.api._v2.keras.callbacks import EarlyStopping
from sklearn.metrics import classification_report, confusion_matrix

import tensorflow as tf
import numpy as np
import os

from bertEncoder import encode_texts, load_encodings
from majority_class import undersample_majority_class

tf.random.set_seed(21)
np.random.seed(38)
os.environ['PYTHONHASHSEED'] = str(13)


name="openai_responses_gpt3_v5"
print("LOADING",name,"EMBEDDINGS")
input_ids_np, attention_masks_np, token_type_ids_np, labels_np = load_encodings(name)


# Split the dataset
dataset = tf.data.Dataset.from_tensor_slices((
    {
        "input_word_ids": input_ids_np,
        "input_mask": attention_masks_np,
        "input_type_ids": token_type_ids_np
    },
    labels_np
))


BATCH_SIZE = 32

total_size = len(labels_np)
dataset, new_length = undersample_majority_class(dataset)
total_size = new_length

# Splitting train_val_data further into training and validation sets
TRAIN_SPLIT_RELATIVE = 0.8  # 80% of TRAIN_VAL_SPLIT for training

train_size = int(total_size * TRAIN_SPLIT_RELATIVE)

train_data = dataset.take(train_size)
val_data = dataset.skip(train_size)

train_data = dataset.take(train_size)
val_data = dataset.skip(train_size)


train_data = train_data.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
val_data = val_data.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

#debug
for _inputs, _labels in train_data.take(1):
  for key, value in _inputs.items():
    print(f"{key}: {value[0].numpy()}")
  print(_labels)

print("=== GET MODEL ===")
model = get_model_with_bert()
print("=== FIT MODEL ===")  

early_stopping = EarlyStopping(
    monitor='loss',
    patience=9,         
    verbose=1,         
    restore_best_weights=True 
)

trained_model = model.fit(
    train_data, 
    epochs=10, 
    validation_data=val_data,
    #callbacks=[early_stopping]
)



test_loss, test_accuracy = model.evaluate(val_data, verbose=2)
print(f"Val accuracy: {test_accuracy*100:.2f}%")
print(f"Val loss: {test_loss}")
model.save(f'models/untrained_bert_fakestack_{name}', save_format='tf')

val_predictions = model.predict(val_data)
print("Pred number values:")
print(val_predictions[:35])
pred_labels = (val_predictions >= 0.5).astype(int).ravel()


true_labels = np.concatenate([labels for _, labels in val_data.as_numpy_iterator()])
print("True values")
print(true_labels[:35])
print("Pred values:")
print(pred_labels[:35])