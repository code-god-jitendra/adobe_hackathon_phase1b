import joblib

class HeadingDetector:
    def __init__(self, model_path="model/heading_model.pkl"):
        self.scaler, self.model = joblib.load(model_path)

    def is_heading(self, block: dict) -> bool:
        font_ratio = block["font_size"] / block["body_font_size"] if block["body_font_size"] else 1.0

        features = [[
            block["font_size"],
            block["is_bold"],
            block["x"],
            block["y"],
            block["char_length"],
            block["body_font_size"],
            font_ratio
        ]]

        X = self.scaler.transform(features)
        return self.model.predict(X)[0] == 1
