from srgnn.model import build_model


def test_model(flat_data, flat_oov):
    # Train with the validation set
    model = build_model(X_val=flat_data, max_epochs=2, k=2)
    model.fit(flat_data)
    model.predict(flat_oov)


def test_model_default_split(flat_data, flat_oov):
    # Test with the default split
    model = build_model(max_epochs=2, k=2)
    model.fit(flat_data)
    model.predict(flat_oov)
