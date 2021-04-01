import pickle
import torch

def limit(a, b):

    a_address = 0
    b_address = 0

    while (a > 100):
        a = a - 100
        a_address += 1

    while (b > 100):
        b = b - 100
        b_address += 1

    return a, b, a_address, b_address




# hardcoded from Grimme's D3
total_records = 161925
num_lines = 32385
records_per_line = 5
with open('c6_unraveled.pkl', 'rb') as f:
    c6_unraveled = pickle.load(f)
    assert len(c6_unraveled) == total_records
    assert int(len(c6_unraveled) / records_per_line) == num_lines

supported_d3_elements = 94
max_references = 5
c6_parameters = torch.zeros((supported_d3_elements, supported_d3_elements,
    max_references, max_references, 3))
max_ci = torch.zeros(supported_d3_elements)

# every line has: 
# 0 1 2 3 4
# C6, a, b, CNa, CNb
# in that order
# a_address and b_address give the conformation's address (?)
# if a or b are greater than 100 this means the conformation address
# has to be moved by +1
for k in range(0, len(c6_unraveled), records_per_line):
    a = int(c6_unraveled[k + 1])
    b = int(c6_unraveled[k + 2])
    a, b, a_address, b_address = limit(a, b)
    a -= 1
    b -= 1
    max_ci[a] = max(max_ci[a], a_address)
    max_ci[b] = max(max_ci[b], b_address)
    # get values for C6 and CNa, CNb
    c6_parameters[a, b, a_address, b_address, 0] = c6_unraveled[k]
    c6_parameters[a, b, a_address, b_address, 1] = c6_unraveled[k + 3]
    c6_parameters[a, b, a_address, b_address, 2] = c6_unraveled[k + 4]
    # get symmetric values
    c6_parameters[b, a, b_address, a_address, 0] = c6_unraveled[k]
    c6_parameters[b, a, b_address, a_address, 1] = c6_unraveled[k + 3]
    c6_parameters[b, a, b_address, a_address, 2] = c6_unraveled[k + 4]
