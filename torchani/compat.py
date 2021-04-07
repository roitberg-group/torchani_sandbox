import sys

if sys.version_info[:2] < (3, 7):

    class FakeFinal:
        def __getitem__(self, x):
            return x

    Final = FakeFinal()
else:
    from torch.jit import Final
