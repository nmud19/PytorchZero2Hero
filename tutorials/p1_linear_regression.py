import torch
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


class LinearRegression(torch.nn.Module):
    """
    This is the custom class to create your module
    """

    def __init__(self, input_dim, output_dim):
        super(LinearRegression, self).__init__()
        self.predict = torch.nn.Linear(
            in_features=input_dim,
            out_features=output_dim
        )

    def forward(self, data):
        preds = self.predict(data)
        return preds


if __name__ == '__main__':
    # Create data
    x, y = load_boston()["data"], load_boston()["target"]

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=100)

    sc = StandardScaler()
    x_train = sc.fit_transform(x_train)
    x_test = sc.transform(x_test)

    x_train = torch.tensor(x_train, dtype=torch.float32)
    x_test = torch.tensor(x_test, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32)

    y_train = y_train.view(y_train.shape[0], 1)
    y_test = y_test.view(y_test.shape[0], 1)

    # Create model
    n_samples, n_features = x.shape
    model = LinearRegression(
        input_dim=n_features,  # no of features
        output_dim=1  # regression output
    )

    # Setup the model
    learning_rate = 0.01
    epochs = 400
    criterion = torch.nn.MSELoss()
    optimiser = torch.optim.SGD(
        model.parameters(),
        lr=learning_rate
    )

    # training loop
    for epoch in range(0, epochs+1):
        # forward pass
        y_hat = model(x_train)

        # loss
        loss = criterion(y_hat, y_train)
        loss.backward()

        # Update the weights
        optimiser.step()
        optimiser.zero_grad()

        if epoch % 40 == 0:
            print(f"epoch {epoch} : loss= {loss.item():.4f}")

    # Test set
    with torch.no_grad():
        y_hat = model(x_test)
        validation_loss = criterion(y_hat, y_test)
        print(f"The MSE on test set is {validation_loss : .4f}")
