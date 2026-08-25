import joblib


def load_model(model_path):
    """
    Load the already-trained final LightGBM regression model.
    """
    return joblib.load(model_path)


def load_selected_features(features_path):
    """
    Load the selected regression features.
    """
    return joblib.load(features_path)