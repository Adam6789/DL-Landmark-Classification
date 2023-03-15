import torch
import torch.nn as nn


# define the CNN architecture
class MyModel(nn.Module):
    def __init__(self, num_classes: int = 1000, dropout: float = 0.7, version: str = "_v3") -> None:

        super().__init__()

        # YOUR CODE HERE
        # Define a CNN architecture. Remember to use the variable num_classes
        # to size appropriately the output of your classifier, and if you use
        # the Dropout layer, use the variable "dropout" to indicate how much
        # to use (like nn.Dropout(p=dropout))
        if version == "_v1":
            self.classifier = nn.Sequential(
                    #(3,224,224)
                    nn.Conv2d(3,16,kernel_size=3,padding='same'),
                    nn.MaxPool2d(2,2),
                    nn.ReLU(),
                    nn.BatchNorm2d(16),
                    nn.Conv2d(16,32,kernel_size=3,padding='same'),
                    nn.MaxPool2d(2,2),
                    nn.ReLU(),
                    nn.BatchNorm2d(32),
                    nn.Conv2d(32,64,kernel_size=3,padding='same'),
                    nn.MaxPool2d(2,2),
                    nn.ReLU(),
                    nn.Dropout(p=dropout),
                    nn.Flatten(),
                    nn.Linear(64*28*28, num_classes),
                    nn.Sigmoid()     
                )
        elif version == "_v_lernt_nicht":
            layers = 4
            factor = ((64*28*28)/(50))**(1/layers)
            self.classifier = nn.Sequential(
                #(3,224,224)
                nn.Conv2d(3,16,kernel_size=3,padding='same'),
                nn.MaxPool2d(2,2),
                nn.ReLU(),
                nn.BatchNorm2d(16),
                nn.Conv2d(16,32,kernel_size=3,padding='same'),
                nn.MaxPool2d(2,2),
                nn.ReLU(),
                nn.BatchNorm2d(32),
                nn.Conv2d(32,64,kernel_size=3,padding='same'),
                nn.MaxPool2d(2,2),
                nn.ReLU(),
#                 nn.BatchNorm2d(64),
#                 nn.Conv2d(64,128,kernel_size=3,padding='same'),
#                 nn.MaxPool2d(2,2),
#                 nn.ReLU(),
                nn.Flatten(),
                nn.Dropout(p=dropout),
                nn.Linear(64*28*28, round(num_classes*factor**3)),
                nn.Dropout(p=dropout),
                nn.ReLU(),
                nn.Linear(round(num_classes*factor**3), round(num_classes*factor**2)),
                nn.Dropout(p=dropout),
                nn.ReLU(),
                nn.Linear(round(num_classes*factor**2), round(num_classes*factor**1)),
                nn.Dropout(p=dropout),
                nn.ReLU(),
                nn.Linear(round(num_classes*factor**1), num_classes),
                nn.Dropout(p=dropout),
                nn.Sigmoid()  
            )
        elif version == "_v2":
            self.classifier = nn.Sequential(
                    #(3,224,224)
                    nn.Conv2d(3,16,kernel_size=3,padding='same'),
                    nn.MaxPool2d(2,2),
                    nn.ReLU(),
                    nn.BatchNorm2d(16),
                    nn.Conv2d(16,32,kernel_size=3,padding='same'),
                    nn.MaxPool2d(2,2),
                    nn.ReLU(),
                    nn.BatchNorm2d(32),
                    nn.Conv2d(32,64,kernel_size=3,padding='same'),
                    nn.MaxPool2d(2,2),
                    nn.ReLU(),
                    nn.Dropout(p=dropout),
                    nn.Flatten(),
                    nn.Linear(64*28*28, 2000),
                    nn.ReLU(),
                    nn.Dropout(p=dropout),
                    nn.Linear(2000, num_classes),
                    nn.Sigmoid(),
                    nn.Dropout(p=dropout),
                )
            



    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # YOUR CODE HERE: process the input tensor through the
        # feature extractor, the pooling and the final linear
        # layers (if appropriate for the architecture chosen)
        x = self.classifier(x)
        return x
    

    


######################################################################################
#                                     TESTS
######################################################################################
import pytest


@pytest.fixture(scope="session")
def data_loaders():
    from .data import get_data_loaders

    return get_data_loaders(batch_size=2)


def test_model_construction(data_loaders):

    model = MyModel(num_classes=23, dropout=0.3)

    dataiter = iter(data_loaders["train"])
    images, labels = dataiter.next()

    out = model(images)

    assert isinstance(
        out, torch.Tensor
    ), "The output of the .forward method should be a Tensor of size ([batch_size], [n_classes])"

    assert out.shape == torch.Size(
        [2, 23]
    ), f"Expected an output tensor of size (2, 23), got {out.shape}"
