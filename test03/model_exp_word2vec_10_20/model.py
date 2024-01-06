from torch import nn


class DNN(nn.Module):
    def __init__(self, vocabulary_size, embedding_dim):
        super(DNN, self).__init__()

        self.embedding = nn.Linear(vocabulary_size, embedding_dim, bias=False)
        print("embedding_size:", list(self.embedding.weight.size()))

        self.layers = nn.Sequential(
            nn.Linear(vocabulary_size * embedding_dim, embedding_dim // 2),
            nn.LeakyReLU(),
            nn.Linear(embedding_dim // 2, 4),
            nn.LeakyReLU(),
            nn.Linear(4, 1),
        )

        # Mean squared error loss
        self.criterion = nn.MSELoss()
        # self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        x = self.embedding(x)
        x = x.view(x.size()[0], -1)
        x = self.layers(x)
        x = x.squeeze(1)
        return x

    def cal_loss(self, pred, target):
        """ Calculate loss """
        return self.criterion(pred, target)
