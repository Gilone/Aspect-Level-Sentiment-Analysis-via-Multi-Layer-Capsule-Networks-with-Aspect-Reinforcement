from keras.models import Model, Sequential
from keras.layers import Input, Embedding, Dropout, Reshape, Dense, LSTM, Multiply, Bidirectional, GRU
from layers import Capsule, Mask, Length, Slice


def Caps_LSTM(embedding_matrix, max_seq_len, embed_dim, num_class, num_routing, use_location=True):
    main_input = Input(shape=(2*max_seq_len,))
    seq_embed = Slice(1, 0, max_seq_len)(main_input)
    seq_embed = Embedding(len(embedding_matrix), embed_dim, input_length=max_seq_len,
                           weights=[embedding_matrix],trainable=False)(seq_embed)
    seq_embed = Dropout(0.3)(seq_embed)
    if use_location:
        seq_weight = Slice(1, max_seq_len, 2 * max_seq_len)(main_input)
        seq_weight = Reshape((-1, 1))(seq_weight)
        lstm = LSTM(300, return_sequences=True, dropout=.5, recurrent_dropout=.5)(seq_embed)
        lstm = LSTM(300, return_sequences=True, dropout=.5, recurrent_dropout=.5)(lstm)
        seq_embed = Multiply()([lstm, seq_weight])
    capsule = Reshape((-1, 50))(seq_embed)
    capsule = Capsule(num_capsule=30, dim_capsule=150, num_routing=num_routing)(capsule)
    capsule = Capsule(num_capsule=num_class, dim_capsule=300, num_routing=num_routing)(capsule)
    length = Length(name='prob')(capsule)
    decoer = Sequential(name='recon')
    decoer.add(Dense(embed_dim, activation='relu', input_dim=300 * num_class))
    mask_input = Input(shape=(num_class,))
    recon_true = decoer(Mask()([capsule, mask_input]))
    recon_false = decoer(Mask(reverse=True)([capsule, mask_input]))
    train_model = Model([main_input, mask_input], [length, recon_true, recon_false])
    recon_top = decoer(Mask(rank=1)(capsule))
    recon_second = decoer(Mask(rank=2)(capsule))
    eval_model = Model([main_input], [length, recon_top, recon_second])
    return train_model, eval_model

