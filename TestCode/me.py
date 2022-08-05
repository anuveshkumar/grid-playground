import torch
import MinkowskiEngine as ME
import matplotlib.pyplot as plt
from common import data_loader


def get_random_coords(dimension=3, tensor_stride=2):
    torch.manual_seed(0)
    # Create random coordinates with tensor stride == 2
    coords = torch.rand(10, dimension + 1)
    coords[:, :dimension] *= 5  # random coords
    coords[:, -1] *= 2  # random batch_index
    coords = ME.utils.sparse_quantize(coords)
    coords[:, :dimension] *= tensor_stride  # make the tensor stride 2
    return coords, tensor_stride


def print_sparse_tensor(tensor):
    for c, f in zip(tensor.C.numpy(), tensor.F.detach().numpy()):
        print(f"Coordinate {c} : Feature {f}")


def conv():
    in_channels, out_channels, D = 5, 3, 2
    coords, feats, labels = data_loader(in_channels, batch_size=1)

    # Convolution
    input = ME.SparseTensor(features=feats, coordinates=coords)
    conv = ME.MinkowskiConvolution(
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        bias=True,
        dimension=D
    )
    output = conv(input)

    print('Input: ')
    print_sparse_tensor(input)

    print('Output:')
    print_sparse_tensor(output)

    # Convolution transpose and generate new coordinates

    strided_coords, tensor_stride = get_random_coords()

    input = ME.SparseTensor(features=torch.rand(len(strided_coords), in_channels),
                            coordinates=strided_coords,
                            tensor_stride=tensor_stride)
    conv_tr = ME.MinkowskiConvolution(
        in_channels,
        out_channels,
        kernel_size=3,
        stride=2,
        bias=True,
        dimension=3
    )
    output = conv_tr(input)
    print('\nInput:')
    print_sparse_tensor(input)

    print('Convolution Transpose Output:')
    print_sparse_tensor(output)


if __name__ == "__main__":
    conv()
