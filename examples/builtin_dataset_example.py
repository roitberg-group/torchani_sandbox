import torchani
# Example of loading builtin datasets builtin datasets come prepackaged with
# torchani.  currently builtin datasets live in moria, and they can be directly
# downloaded through torchani.

# This will download the corresponding dataset int './datasets/ani1x/' and
# perform a checksum to make sure the file's integrity is not compromised. If
# the function is called again and the dataset is already on the path, only the
# checksum is performed, the data is not downloaded.
ds = torchani.datasets.ANI1x('./datasets/ani1x/', download=True)
# This constructs an ANIDataset class, so you can iterate over it or
# perform whatever operation you want.
for k, v in ds.items():
    print(k, v)
# The comp6v1 and 2x datasets are also available.
# (watch out, downloading the datasets may take some time)
ds = torchani.datasets.COMP6v1('./datasets/comp6v1/', download=True)
ds = torchani.datasets.ANI2x('./datasets/ani2x/', download=True)
