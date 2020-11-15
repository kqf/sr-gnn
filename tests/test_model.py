from srgnn.model import build_model


def test_model(flat_data, flat_oov):
    model = build_model(max_epochs=2, k=2)
    model.fit(flat_data)
    model.predict(flat_oov)
