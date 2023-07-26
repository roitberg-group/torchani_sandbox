# requirements:
"""
pip install kaleido
mamba install -c conda-forge nbformat
pip install plotly
"""

import torch
import torchani
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from plotly.subplots import make_subplots
import plotly.graph_objects as go

# Define the number of points
num_points = 30
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
aev_computer = torchani.AEVComputer4Body.like_1x().to(device)

# atom_i, atom_j
atom_i = torch.tensor([0, 0, 0], dtype=torch.float32).to(device)
atom_j = torch.tensor([-1, 0, 0], dtype=torch.float32).to(device)


def create_G_max(atom_k):
    # atom_l
    x = torch.linspace(-3.5, 3.5, num_points).to(device)
    y = torch.linspace(-3.5, 3.5, num_points).to(device)
    z = torch.linspace(-3.5, 3.5, num_points).to(device)
    X, Y, Z = torch.meshgrid(x, y, z, indexing='ij')
    atom_l = torch.stack((X.reshape(-1), Y.reshape(-1), Z.reshape(-1)), dim=1)  # [num_points**3, 3]
    atom_i_ = atom_i.broadcast_to(atom_l.shape)  # [num_points**3, 3]
    atom_j_ = atom_j.broadcast_to(atom_l.shape)  # [num_points**3, 3]
    atom_k = atom_k.broadcast_to(atom_l.shape)  # [num_points**3, 3]
    coordinates = torch.cat((atom_i_.unsqueeze(1), atom_j_.unsqueeze(1), atom_k.unsqueeze(1), atom_l.unsqueeze(1)), dim=1)  # [num_points**3, 4, 3]
    species = torch.tensor([1, 1, 1, 1], dtype=torch.int64).to(device).unsqueeze(0).expand(coordinates.shape[0], -1)  # [num_points**3, 4]

    aevs = aev_computer((species, coordinates))[1]
    fourbody_aevs_atom_i = aevs[:, 0, -aev_computer.fourbody_length:]
    G_max = fourbody_aevs_atom_i.max(dim=1).values.cpu().detach()

    return atom_l, G_max


# Create a 3x3 subplot layout
theta_jiks = np.linspace(1 / 6 * np.pi, 1.0 * np.pi, 6)
fig = make_subplots(rows=2, cols=3, subplot_titles=[f'Theta_jik = {theta/np.pi * 180:.1f}°' for theta in theta_jiks], specs=[[{'type': 'volume'}] * 3] * 2)

# first run to find the max value to set the colorbar, yes, it is stupid and wasting computation..
isomax = 0.0
for i, theta_jik in enumerate(theta_jiks):
    atom_k = torch.tensor([np.cos(np.pi + theta_jik), np.sin(np.pi + theta_jik), 0], dtype=torch.float32).to(device)
    atom_l, G_max = create_G_max(atom_k)
    isomax = max(isomax, G_max.max().item())

# the real calculations and plot
for i, theta_jik in enumerate(theta_jiks):
    atom_k = torch.tensor([np.cos(np.pi + theta_jik), np.sin(np.pi + theta_jik), 0], dtype=torch.float32).to(device)
    atom_l, G_max = create_G_max(atom_k)
    X, Y, Z = atom_l[:, 0].cpu(), atom_l[:, 1].cpu(), atom_l[:, 2].cpu()
    values = c = G_max[:].cpu()

    row = i // 3 + 1  # Calculate the row index for the subplot
    col = i % 3 + 1  # Calculate the column index for the subplot

    # Add the volume to the subplot
    fig.add_trace(go.Volume(
        x=X.flatten().cpu(),
        y=Y.flatten().cpu(),
        z=Z.flatten().cpu(),
        value=values.flatten().cpu(),
        isomin=2e-3,
        isomax=isomax,
        cmin=0,
        opacity=0.15,
        surface_count=30,
        showscale=(i == len(theta_jiks) - 1),  # Hide the color scale for each subplot
        name=f'Theta_jik = {theta_jik/np.pi * 180:.1f}°'  # Set a name for the subplot
    ), row=row, col=col)

    # Add the vectors to the subplot
    fig.add_trace(go.Scatter3d(x=[atom_i[0].cpu(), atom_j[0].cpu() * 4], y=[atom_i[1].cpu(), atom_j[1].cpu() * 4], z=[atom_i[2].cpu(), atom_j[2].cpu() * 4], mode='lines', line=dict(width=5, color="red"), showlegend=False), row=row, col=col)
    fig.add_trace(go.Scatter3d(x=[atom_i[0].cpu(), atom_k[0].cpu() * 4], y=[atom_i[1].cpu(), atom_k[1].cpu() * 4], z=[atom_i[2].cpu(), atom_k[2].cpu() * 4], mode='lines', line=dict(width=5, color="green"), showlegend=False), row=row, col=col)

fig.update_layout(width=900, height=600, title_text="4-body Gmax for different theta_jiks")

for i in range(len(theta_jiks)):
    fig.layout[f'scene{i+1}'].update(camera=dict(eye=dict(x=1.5, y=-1.5, z=0.9)))

# fig.show(config=dict(toImageButtonOptions=dict(scale=6)))
fig.write_image('4-body_Gmax.png', scale=6)
