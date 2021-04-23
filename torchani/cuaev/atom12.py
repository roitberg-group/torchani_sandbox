import torch
import timeit

from torch.serialization import save
device = 'cuda'
# device = 'cpu'

# save
# atom_index21_f = atom_index12.flip(0).view(-1)
# atom_index12_f = atom_index12.view(-1)
# atomI, idxs = atom_index12_f.sort()

# rev_idxs = idxs % atom_index12.shape[1]
# atomJ = atom_index21_f.index_select(0, idxs)
# sign = ((atomI < atomJ).int() - 0.5) * 2
# diff_vector = diff_vector.index_select(0, rev_idxs) * sign.view(-1, 1)
# distances = distances.index_select(0, rev_idxs)
# atom_index12 = torch.cat([atomI, atomJ]).view(2, -1)
# print(atom_index12.shape)
# print(torch.unique(atom_index12[0]).shape)
atom_index12 = torch.tensor([[0, 0, 0, 1, 1, 2],
                            [1, 2, 3, 2, 3, 3]], device=device)
shift_values = torch.tensor([[1.0, 0.0, 0.0],
                            [2.0, 0.0, 0.0],
                            [3.0, 0.0, 0.0],
                            [4.0, 0.0, 0.0],
                            [5.0, 0.0, 0.0],
                            [6.0, 0.0, 0.0]], device=device)
atom_index12 = torch.triu_indices(7000, 7000, 1, device=device)
shift_values = torch.ones(atom_index12.shape[1], 3, device=device)

def funca():
    torch.cuda.synchronize()
    atom_index12_f = atom_index12.view(-1)
    # replace with cub sort
    atomI, idxs = atom_index12_f.sort()
    # rev_idxs = torch.arange(0, atom_index12.shape[1], device=device).repeat(2)
    atom_index21_f = atom_index12.flip(0).view(-1)
    # rev_idxs = rev_idxs.index_select(0, idxs)
    rev_idxs = idxs % atom_index12.shape[1]
    atomJ = atom_index21_f.index_select(0, idxs)
    # sign = 
    shift = shift_values.index_select(0, rev_idxs)
    # print(atom_index12)
    # print(shift_values)
    # print(atomI)
    # print(atomJ)
    # print(idxs)
    # print(rev_idxs)
    # print(shift)
    # print()
    torch.cuda.synchronize()

def funcb():
    torch.cuda.synchronize()
    atom_index12_ = torch.cat((atom_index12, atom_index12.clone().flip(0)), dim=-1)
    shift_values_ = shift_values.repeat(2, 1)
    aug = atom_index12_[0, :] * atom_index12_.max() + atom_index12_[1, :]
    idxs = aug.sort().indices
    atom_index12_ = atom_index12_.index_select(1, idxs)
    shift_values_ = shift_values_.index_select(0, idxs)
    # print(idxs)
    # print(atom_index12)
    # print(shift_values)
    torch.cuda.synchronize()

print(timeit.timeit(stmt=funca, number=10)/10)
print(timeit.timeit(stmt=funcb, number=10)/10)
funca()