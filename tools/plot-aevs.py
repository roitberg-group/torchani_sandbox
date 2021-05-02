import torch
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa
from matplotlib import cm
from torchani.aev.aev_terms import StandardAngular, StandardRadial, PhysNetRadial

if __name__ == '__main__':
    device = torch.device('cuda')

    angular_terms = StandardAngular.like_1x().to(device)

    v1 = torch.tensor([0.0, 0.0, 0.0], device=device)

    size_r = 100
    size_theta = 100
    r = torch.linspace(0, angular_terms.cutoff, size_r, device=device).view(-1, 1, 1)
    theta = torch.linspace(0, math.pi, size_theta, device=device).view(1, -1, 1)

    v1 = torch.tensor([1.0, 0.0, 0.0], device=device).view(1, 1, 3).repeat(1, size_theta, 1) * r
    v1 = v1.view(-1, 3)
    v2 = torch.cat((torch.cos(theta), torch.sin(theta), torch.zeros((1, size_theta, 1), dtype=theta.dtype, device=device)), dim=-1) * r
    v2 = v2.view(-1, 3)

    X, Y, Z = v2.unbind(-1)
    X = X.view(size_r, size_theta).cpu().numpy()
    Y = Y.view(size_r, size_theta).cpu().numpy()

    vectors12 = torch.cat((v1.unsqueeze(0), v2.unsqueeze(0)), dim=0)

    # plot standard ani1x angular AEV
    angular_aev = angular_terms(vectors12).view(size_r, size_theta, -1).permute(2, 0, 1)
    upper_aev = angular_aev.max(0)[0]

    fig1 = plt.figure()
    ax = fig1.gca(projection='3d')
    ax.plot_surface(X, Y, upper_aev.cpu().numpy(), cmap=cm.jet, linewidth=0, antialiased=True)
    ax.set_title('ANI 1x angular upper aev')
    ax.set_xlim(-angular_terms.cutoff, angular_terms.cutoff)
    ax.set_ylim(-angular_terms.cutoff, angular_terms.cutoff)
    plt.show()

    # plot standard ani1x radial AEV
    radial_terms = StandardRadial.like_1x().to(device)
    size_radial_r = 1000
    distances = torch.linspace(0, radial_terms.cutoff, size_radial_r, device=device)
    radial_aev = radial_terms(distances).permute(1, 0)
    cutoff_envelope = radial_terms.cutoff_fn(distances, radial_terms.cutoff)

    fig1, ax1 = plt.subplots()
    max_values_radial = []
    for term in radial_aev:
        ax1.plot(distances.cpu(), term.cpu())
        ax1.set_xlabel(r'Distance ($\AA$)')
        ax1.set_ylabel(r'AEV term intensity')
        ax1.set_title('ANI 1x radial aevs')
        max_values_radial.append(term.max().item())

    # cutoff envelope for 1x is actually multiplied by 0.25
    ax1.plot(distances.cpu(), 0.25 * cutoff_envelope.cpu(), color='k', linestyle='dashed')
    plt.show(block=False)

    # plot physnet radial AEV, this reproduces figure 3 from their paper note
    # that their AEV form doesn't make a lot of sense, since the terms are
    # incredibly close together and they have a very large number of terms in a
    # super small range of distances close to 1 angstrom
    radial_terms = PhysNetRadial().to(device)
    size_radial_r = 1000
    distances = torch.linspace(0, radial_terms.cutoff, size_radial_r, device=device)
    radial_aev = radial_terms(distances).permute(1, 0)
    cutoff_envelope = radial_terms.cutoff_fn(distances, radial_terms.cutoff)

    fig1, ax = plt.subplots()
    max_values_radial = []
    for term in radial_aev:
        ax.plot(distances.cpu(), term.cpu())
        ax.set_xlabel(r'Distance ($\AA$)')
        ax.set_ylabel(r'AEV term intensity')
        ax.set_title('PhysNet radial aevs')
        max_values_radial.append(term.max().item())
    ax.plot(distances.cpu(), cutoff_envelope.cpu(), color='k', linestyle='dashed')
    plt.show()
