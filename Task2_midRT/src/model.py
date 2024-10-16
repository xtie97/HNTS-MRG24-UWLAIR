from .segresnet_ds import MASegResNetDS
import torch


def get_network(in_channels=3, n_class=3):
    model = MASegResNetDS(
        init_filters=32,
        blocks_down=[1, 2, 2, 4, 4, 4],
        norm="INSTANCE",
        in_channels=in_channels,
        out_channels=n_class,
        dsdepth=4,
    )
    # channels: midRT, preRT mask class1, preRT mask class2
    return model


if __name__ == "__main__":
    model = get_network(in_channels=3).to("cuda")
    # create a random input tensor

    x = torch.rand((1, 3, 192, 192, 128)).to("cuda")
    # 192, 96, 48, 24, 12, 6
    out = model(x)

    print("Model loaded successfully")
    print("Done")
