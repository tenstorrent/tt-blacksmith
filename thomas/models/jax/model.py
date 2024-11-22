import flax.linen as nn


class MLP(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = x.reshape((x.shape[0], -1))
        x = nn.Dense(features=256)(x)
        x = nn.relu(x)
        x = nn.Dense(features=256)(x)
        x = nn.relu(x)
        x = nn.Dense(features=10)(x)
        return x


class Models:
    def __init__(self, model_type="MLP"):
        if model_type == "MLP":
            self.model = MLP()
        # Add other model types here
        # elif model_type == 'CNN':
        #     self.model = CNN()
        # elif model_type == 'LLM':
        #     self.model = LLM()
        else:
            raise ValueError(f"Unknown model type: {model_type}")

    def __call__(self, x):
        return self.model(x)
