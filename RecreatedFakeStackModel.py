from keras.api._v2.keras.models import Model
from keras.api._v2.keras.layers import Input, Conv1D, MaxPooling1D, Dropout, add, Dense, LSTM, UpSampling1D
from keras.api._v2.keras.optimizers import Adam
import tensorflow as tf
import tensorflow_hub as hub


def get_model_with_bert():
    
    bert_model_url = "https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-2_H-128_A-2/2"
    bert_layer = hub.KerasLayer(bert_model_url, trainable=False, output_key='sequence_output')

    max_len = 128 

    input_word_ids = Input(shape=(max_len,), dtype=tf.int32, name="input_word_ids")
    input_mask = Input(shape=(max_len,), dtype=tf.int32, name="input_mask")
    input_type_ids = Input(shape=(max_len,), dtype=tf.int32, name="input_type_ids")

    bert_outputs = bert_layer({
        'input_word_ids': input_word_ids,
        'input_mask': input_mask,
        'input_type_ids': input_type_ids
    })
    
    # Initial Dropout layer to counter overfitting
    x = Dropout(0.2)(bert_outputs)

    # First Conv1D + ReLU + MaxPooling1D
    conv1 = Conv1D(filters=64, kernel_size=5, activation='relu', padding='same')(x)
    print(f"conv1 shape: {conv1.shape}")
    pool1 = MaxPooling1D(pool_size=4)(conv1)
    print(f"pool1 shape: {pool1.shape}")
    # Second Conv1D + ReLU + MaxPooling1D
    conv2 = Conv1D(filters=32, kernel_size=5, activation='relu', padding='same')(pool1)
    
    print(f"conv2 shape: {conv2.shape}")
    
    # UpSampling conv2 to match conv1 dimensions if necessary
    conv2_upscaaled = UpSampling1D(size=4)(conv2)
    print(f"conv2_upscaaled shape: {conv2_upscaaled.shape}")
    # Adjusting conv1 to match conv2 dimensions if necessary
    conv1_adjusted = Conv1D(filters=32, kernel_size=5, activation='relu', padding='same')(conv1)

    print(f"conv1_adjusted shape: {conv1_adjusted.shape}")
    # Adding skip connection 1
    skip1 = add([conv2_upscaaled, conv1_adjusted])
    pool2 = MaxPooling1D(pool_size=4)(skip1)

    # Third Conv1D + ReLU + MaxPooling1D
    conv3 = Conv1D(filters=32, kernel_size=5, activation='relu', padding='same')(pool2)

    # Adjusting conv2 to match conv3 dimensions if necessary
    conv2_adjusted = Conv1D(filters=32, kernel_size=5, activation='relu', padding='same')(conv2)

    # Adding skip connection 2
    skip2 = add([conv3, conv2_adjusted])
    pool3 = MaxPooling1D(pool_size=4)(skip2)
    

    # Fourth Conv1D + ReLU + MaxPooling1D
    conv4 = Conv1D(filters=16, kernel_size=3, activation='relu', padding='same')(pool3)
    pool4 = MaxPooling1D(pool_size=2)(conv4)

    # Fifth Conv1D + ReLU + MaxPooling1D
    conv5 = Conv1D(filters=8, kernel_size=3, activation='relu', padding='same')(pool4)
    pool5 = MaxPooling1D(pool_size=2)(conv5)

    # Instead of flattening and going to a Dense layer directly,
    # feed the output of the last pooling layer to an LSTM layer
    lstm_out = LSTM(64, return_sequences=False)(pool5)  # 64 units in LSTM layer

    # Dense layer for classification 
    dense_out = Dense(1, activation='sigmoid')(lstm_out)

    model = Model(inputs=[input_word_ids, input_mask, input_type_ids], outputs=dense_out)

    model.summary()
    learning_rate = 0.001
    optimizer = Adam(learning_rate=learning_rate)

    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

    return model

