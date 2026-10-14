def encode_state(features) -> list[float]:
    encoded_features = []
    for feature in features:
        if isinstance(feature, tuple) or isinstance(feature, list):
            encoded_features.extend(encode_state(feature))
        else:
            encoded_features.append(float(feature))
    return encoded_features


features = (2, (-1, 0), 1, (1, 1), False)
print(encode_state(features))