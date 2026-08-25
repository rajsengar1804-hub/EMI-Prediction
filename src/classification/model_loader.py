import joblib


def load_model(model_path):
    """
    Load the already-trained XGBoost classification model.
    """
    return joblib.load(model_path)


def load_label_encoder(encoder_path):
    """
    Load the label encoder used during training.
    """
    return joblib.load(encoder_path)


def load_selected_features(features_path):
    """
    Load the features selected during training.
    """
    return joblib.load(features_path)