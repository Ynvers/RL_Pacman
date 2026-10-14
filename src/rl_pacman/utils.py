from collections.abc import Iterable


def encode_state(features: Iterable[object]) -> list[float]:
    encoded_features: list[float] = []
    for feature in features:
        if isinstance(feature, (tuple, list)):
            encoded_features.extend(encode_state(feature))
        else:
            encoded_features.append(float(feature))
    return encoded_features


if __name__ == "__main__":
    features = (2, (-1, 0), 1, (1, 1), False)
    print(encode_state(features))