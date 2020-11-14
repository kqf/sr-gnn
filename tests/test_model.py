from srgnn.model import build_model


def test_model(flat_data, flat_oov):
    model = build_model()
    model.fit(flat_data)
    model.predict(flat_oov)
