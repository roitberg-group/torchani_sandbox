#include <aev.h>
#include <thrust/equal.h>
#include <torch/extension.h>
#include <cub/cub.cuh>
#include <iostream>
#include <vector>

#include <ATen/Context.h>
#include <THC/THC.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include <THC/THCThrustAllocator.cuh>

#define PI 3.141592653589793
using torch::Tensor;

#ifdef DEBUG
#define DEBUG_TEST 1
#else
#define DEBUG_TEST 0
#endif

// fetch from the following matrix
// [[ 0,  1,  2,  3,  4],
//  [ 1,  5,  6,  7,  8],
//  [ 2,  6,  9, 10, 11],
//  [ 3,  7, 10, 12, 13],
//  [ 4,  8, 11, 13, 14]]
constexpr int csubaev_offsets(int i, int j, int n) {
  int larger = std::max(i, j);
  int smaller = std::min(i, j);
  int starting = smaller * (2 * n - smaller + 1) / 2; // n + (n - 1) + ... + (n - smaller + 1)
  int offset = larger - smaller;
  return starting + offset;
}

// convert pair index to reversed j index
// e.g. convert following indices
// [ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14]
// to j:
// [ 1,  2,  2,  3,  3,  3,  4,  4,  4,  4,  5,  5,  5,  5,  5]
// then k will be:
// [ 0,  0,  1,  0,  1,  2,  0,  1,  2,  3,  0,  1,  2,  3,  4]
constexpr int pairidx_to_j(int n) {
  int j = ceil((sqrt(8 * (n + 1) + 1.f) - 1) / 2.f); // x (x + 1) / 2 = n --> x = (-b + sqrt(1 + 8n)) / 2
  return j;
}

// used to group Rijs by atom id
__host__ __device__ bool operator==(const PairDist& lhs, const PairDist& rhs) {
  return lhs.midx == rhs.midx && lhs.i == rhs.i;
}

/// Alignment of memory. Must be a power of two
/// \tparam boundary Boundary to align to (NOTE: must be power of 2)
/// \param value Input value that is to be aligned
/// \return Value aligned to boundary
template <int32_t boundary>
__host__ __device__ __forceinline__ int align(const int& value) {
  static_assert((boundary & (boundary - 1)) == 0, "Boundary for align must be power of 2");
  return (value + boundary) & ~(boundary - 1);
}

template <typename SpeciesT, typename DataT, typename IndexT = int>
__global__ void pairwiseDistance(
    const torch::PackedTensorAccessor32<SpeciesT, 2, torch::RestrictPtrTraits> species_t,
    const torch::PackedTensorAccessor32<DataT, 3, torch::RestrictPtrTraits> pos_t,
    torch::PackedTensorAccessor32<SpeciesT, 1, torch::RestrictPtrTraits> radialNumPairsPerAtom_t,
    AtomI* __restrict__ atom_i,
    int* __restrict__ atomJ_p,
    float* __restrict__ distJ_p,
    const DataT Rcr,
    const IndexT max_natoms_per_mol) {
  extern __shared__ float smem[];
  int* s_pcounter_i = reinterpret_cast<int*>(&smem[0]);
  int* s_type = reinterpret_cast<int*>(&smem[max_natoms_per_mol]);
  float3* s_pos = reinterpret_cast<float3*>(&smem[max_natoms_per_mol * 2]);

  const float3* pos_t_3 = reinterpret_cast<const float3*>(&pos_t[0][0][0]);

  int mol_idx = blockIdx.x;
  int tidx = blockDim.x * threadIdx.y + threadIdx.x;

  for (int i = tidx; i < max_natoms_per_mol; i += blockDim.x * blockDim.y) {
    SpeciesT type_i = species_t[mol_idx][i];
    s_type[i] = type_i;
    s_pcounter_i[i] = 0;
    if (type_i != -1) {
      s_pos[i] = pos_t_3[max_natoms_per_mol * mol_idx + i];
    }
  }
  __syncthreads();

  int pairs_per_mol = max_natoms_per_mol * (max_natoms_per_mol - 1);

  for (int i = threadIdx.y; i < max_natoms_per_mol; i += blockDim.y) {
    SpeciesT type_i = s_type[i];
    if (type_i != -1) {
      float3 pos_i = s_pos[i];

      for (int j = threadIdx.x; j < max_natoms_per_mol; j += blockDim.x) {
        SpeciesT type_j = s_type[j];
        if (type_j != -1 && i != j) {
          float3 delta = make_float3(s_pos[j].x - pos_i.x, s_pos[j].y - pos_i.y, s_pos[j].z - pos_i.z);
          DataT Rsq = delta.x * delta.x + delta.y * delta.y + delta.z * delta.z;
          DataT Rij = sqrt(Rsq);
          if (Rij <= Rcr) {
            int pidx = atomicAdd(&s_pcounter_i[i], 1);
            atomJ_p[mol_idx * pairs_per_mol + i * (max_natoms_per_mol - 1) + pidx] = j;
            distJ_p[mol_idx * pairs_per_mol + i * (max_natoms_per_mol - 1) + pidx] = Rij;
            // printf("i %d, j %d, pidx %d, Rij %f\n", i, j, pidx, Rij);
          } // if Rij is within Rcr
        } // if j is not padding atom and i is not j
      }
    } // if i is not padding atom
    AtomI aI = {mol_idx, i};
    atom_i[mol_idx * max_natoms_per_mol + i] = aI;
  }
  __syncthreads();
  for (int i = tidx; i < max_natoms_per_mol; i += blockDim.x * blockDim.y) {
    // printf("i %d, pidx %d\n", i, s_pcounter_i[i]);
    radialNumPairsPerAtom_t[mol_idx * max_natoms_per_mol + i] = s_pcounter_i[i];
  }
}

// ATOM_J_PER_TILE stands for a tile of atoms J that are loaded into share memory
// ATOM_J_PER_SUBTILE is the tile of atoms J that are really parallel calculating
template <int ATOM_I_PER_BLOCK, int ATOM_J_PER_TILE, typename SpeciesT, typename DataT, typename IndexT = int>
__global__ void pairwiseDistanceSingleMolecule(
    const torch::PackedTensorAccessor32<SpeciesT, 2, torch::RestrictPtrTraits> species_t,
    const torch::PackedTensorAccessor32<DataT, 3, torch::RestrictPtrTraits> pos_t,
    torch::PackedTensorAccessor32<SpeciesT, 1, torch::RestrictPtrTraits> radialNumPairsPerAtom_t,
    AtomI* __restrict__ atom_i,
    int* __restrict__ atomJ_p,
    float* __restrict__ distJ_p,
    const DataT Rcr,
    const IndexT max_natoms_per_mol) {
  __shared__ int s_pcounter_i[ATOM_I_PER_BLOCK];
  __shared__ float3 s_coord_j[ATOM_J_PER_TILE];
  const float3* pos_t_3 = reinterpret_cast<const float3*>(&pos_t[0][0][0]);

  constexpr int mol_idx = 0;
  int natom_pairs = max_natoms_per_mol * (max_natoms_per_mol - 1);
  int i = blockIdx.x * blockDim.y + threadIdx.y;
  int ii = threadIdx.y;
  int sidx = blockDim.x * threadIdx.y + threadIdx.x;
  int num_tiles = (max_natoms_per_mol + ATOM_J_PER_TILE - 1) / ATOM_J_PER_TILE;

  // i >= max_natoms_per_mol is still needed to load share memory for j
  SpeciesT type_i;
  float3 coord_i;
  if (i < max_natoms_per_mol) {
    type_i = species_t[mol_idx][i];
    coord_i = pos_t_3[mol_idx * max_natoms_per_mol + i];
  }

  if (sidx < ATOM_I_PER_BLOCK)
    s_pcounter_i[sidx] = 0;
  __syncthreads();

  for (int tileidx = 0; tileidx < num_tiles; tileidx++) {
    // load 1024 atoms j into share memory
    int jidx = ATOM_J_PER_TILE * tileidx + sidx;
    if (jidx < max_natoms_per_mol) {
      // TODO Test this is coalescing
      s_coord_j[sidx] = pos_t_3[max_natoms_per_mol * mol_idx + jidx];
    }

    __syncthreads();
    for (int jj = threadIdx.x; jj < ATOM_J_PER_TILE && i < max_natoms_per_mol; jj += blockDim.x) {
      int j = jj + ATOM_J_PER_TILE * tileidx;
      if (j < max_natoms_per_mol) {
        float3 delta =
            make_float3(s_coord_j[jj].x - coord_i.x, s_coord_j[jj].y - coord_i.y, s_coord_j[jj].z - coord_i.z);
        SpeciesT type_j = species_t[mol_idx][j];
        DataT Rsq = delta.x * delta.x + delta.y * delta.y + delta.z * delta.z;
        if (type_i != -1 && type_j != -1 && i != j) {
          DataT Rij = sqrt(Rsq);
          if (Rij <= Rcr) {
            int pidx = atomicAdd(&s_pcounter_i[ii], 1);
            atomJ_p[mol_idx * natom_pairs + i * (max_natoms_per_mol - 1) + pidx] = j;
            distJ_p[mol_idx * natom_pairs + i * (max_natoms_per_mol - 1) + pidx] = Rij;
            // printf("i %d, j %d, pidx %d, Rij %f\n", i, j, pidx, Rij);
          }
        }
      }
    }
    __syncthreads();
  }

  i = sidx + blockIdx.x * blockDim.y;
  if (sidx < ATOM_I_PER_BLOCK && i < max_natoms_per_mol) {
    radialNumPairsPerAtom_t[i] = s_pcounter_i[sidx];
    AtomI aI = {mol_idx, i};
    atom_i[mol_idx * max_natoms_per_mol + i] = aI;
  }
}

// every block compute blocksize RIJ's gradient by column major, to avoid atomicAdd waiting
template <bool is_double_backward, typename DataT, typename IndexT = int>
__global__ void pairwiseDistance_backward_or_doublebackward(
    torch::PackedTensorAccessor32<DataT, 3, torch::RestrictPtrTraits> pos_t,
    torch::PackedTensorAccessor32<DataT, 1, torch::RestrictPtrTraits>
        grad_dist, // ddist for backward, dddist for double backward
    torch::PackedTensorAccessor32<DataT, 3, torch::RestrictPtrTraits>
        grad_coord_or_force, // dcoord for backward, dforce(i.e. ddcoord) for double backward
    const PairDist* d_radialRij,
    IndexT nRadialRij) {
  int gidx = threadIdx.x * gridDim.x + blockIdx.x;

  if (gidx >= nRadialRij)
    return;

  PairDist d = d_radialRij[gidx];
  DataT Rij = d.Rij;
  int mol_idx = d.midx;
  int i = d.i;
  int j = d.j;

  const DataT delx = pos_t[mol_idx][j][0] - pos_t[mol_idx][i][0];
  const DataT dely = pos_t[mol_idx][j][1] - pos_t[mol_idx][i][1];
  const DataT delz = pos_t[mol_idx][j][2] - pos_t[mol_idx][i][2];

  if (is_double_backward) {
    auto& grad_force = grad_coord_or_force;
    DataT grad_force_coord_Rij_item = (grad_force[mol_idx][j][0] - grad_force[mol_idx][i][0]) * delx / Rij +
        (grad_force[mol_idx][j][1] - grad_force[mol_idx][i][1]) * dely / Rij +
        (grad_force[mol_idx][j][2] - grad_force[mol_idx][i][2]) * delz / Rij;

    grad_dist[gidx] = grad_force_coord_Rij_item;
  } else {
    auto& grad_coord = grad_coord_or_force;

    DataT grad_dist_coord_x = delx / Rij;
    DataT grad_dist_coord_y = dely / Rij;
    DataT grad_dist_coord_z = delz / Rij;
    DataT grad_radial_dist_item = grad_dist[gidx];

    atomicAdd(&grad_coord[mol_idx][j][0], grad_radial_dist_item * grad_dist_coord_x);
    atomicAdd(&grad_coord[mol_idx][j][1], grad_radial_dist_item * grad_dist_coord_y);
    atomicAdd(&grad_coord[mol_idx][j][2], grad_radial_dist_item * grad_dist_coord_z);
    atomicAdd(&grad_coord[mol_idx][i][0], -grad_radial_dist_item * grad_dist_coord_x);
    atomicAdd(&grad_coord[mol_idx][i][1], -grad_radial_dist_item * grad_dist_coord_y);
    atomicAdd(&grad_coord[mol_idx][i][2], -grad_radial_dist_item * grad_dist_coord_z);
  }
}

// TODO: TILEX is 8 for ANI1x, 4 for ANI2x
template <
    int BLOCK_X,
    int BLOCK_Y,
    typename SpeciesT,
    typename DataT,
    typename IndexT = int,
    int TILEX = 4,
    int TILEY = 8>
__global__ void cuAngularAEVs(
    const torch::PackedTensorAccessor32<SpeciesT, 2, torch::RestrictPtrTraits> species_t,
    const torch::PackedTensorAccessor32<DataT, 3, torch::RestrictPtrTraits> pos_t,
    const torch::PackedTensorAccessor32<DataT, 1, torch::RestrictPtrTraits> ShfA_t,
    const torch::PackedTensorAccessor32<DataT, 1, torch::RestrictPtrTraits> ShfZ_t,
    const torch::PackedTensorAccessor32<DataT, 1, torch::RestrictPtrTraits> EtaA_t,
    const torch::PackedTensorAccessor32<DataT, 1, torch::RestrictPtrTraits> Zeta_t,
    torch::PackedTensorAccessor32<DataT, 3, torch::RestrictPtrTraits> aev_t,
    const int* __restrict__ atomJ,
    const float* __restrict__ distJ,
    const AtomI* __restrict__ atom_i,
    const int* __restrict__ d_nPairsPerCenterAtom,
    const int* __restrict__ d_centerAtomStartIdx,
    float Rca,
    int angular_length,
    int angular_sublength,
    int radial_length,
    int num_species,
    int maxnbrs_per_atom_aligned,
    int angular_length_aligned, // TODO Remove this
    int ncentral_atoms) {
  constexpr int BLOCK_SIZE = BLOCK_X * BLOCK_Y;

  extern __shared__ DataT smem[];
  __shared__ float s_theta[BLOCK_SIZE];

  int cIdx = blockIdx.x; // central atom id
  int tIdx = threadIdx.y * blockDim.x + threadIdx.x; // local thread idx
  const float3* pos_t_3 = reinterpret_cast<const float3*>(&pos_t[0][0][0]);
  const int max_natoms_per_mol = pos_t.size(1);
  int jnum = d_nPairsPerCenterAtom[cIdx];

  if (cIdx >= ncentral_atoms)
    return;
  if (jnum < 2)
    return;

  int laneIdx = threadIdx.x;

  DataT* saev = &smem[0];

  int offset = angular_length_aligned;
  float3* svec = reinterpret_cast<float3*>(&smem[offset]);

  offset += 3 * maxnbrs_per_atom_aligned;
  DataT* sdist = &smem[offset];

  offset += maxnbrs_per_atom_aligned;
  DataT* sfc = &smem[offset];

  offset += maxnbrs_per_atom_aligned;
  int* stype = (int*)&smem[offset];

  DataT EtaA = EtaA_t[0];
  DataT Zeta = Zeta_t[0];

  IndexT nShfA = ShfA_t.size(0);
  IndexT nShfZ = ShfZ_t.size(0);

  int start_idx = d_centerAtomStartIdx[cIdx];
  int totalpairs = jnum * (jnum - 1) / 2;
  AtomI aI = atom_i[cIdx];
  int mol_idx = aI.midx;
  int i = aI.i;

  for (int iaev = tIdx; iaev < angular_length; iaev += blockDim.x * blockDim.y) {
    saev[iaev] = 0;
  }

  float3 coord_i = pos_t_3[mol_idx * max_natoms_per_mol + i];

  for (int jj = tIdx; jj < jnum; jj += blockDim.x * blockDim.y) {
    DataT Rij = distJ[start_idx + jj];
    int j = atomJ[start_idx + jj];
    SpeciesT type_j = species_t[mol_idx][j];
    float3 coord_j = pos_t_3[mol_idx * max_natoms_per_mol + j]; // memory coalessing is not a big deal here, but need 7
                                                                // more registers
    // SpeciesT type_j = 1;
    // float3 coord_j = make_float3(1.0, 1.0, 1.0);
    svec[jj] = make_float3(coord_j.x - coord_i.x, coord_j.y - coord_i.y, coord_j.z - coord_i.z);
    stype[jj] = type_j;
    sdist[jj] = Rij;
    DataT fc_ij = 0.5 * __cosf(PI * Rij / Rca) + 0.5; // cos() increase registers from 32 to 45, __cosf() to 38
    // DataT fc_ij = 0.5;
    sfc[jj] = fc_ij;
  }
  __syncthreads();

  short2 tile = make_short2(laneIdx % TILEX, laneIdx / TILEX);

  for (int n = threadIdx.y; n < ((totalpairs + BLOCK_SIZE - 1) / BLOCK_SIZE) * BLOCK_SIZE; n += blockDim.y) {
    // store 1 blocksize of theta to share mem
    if (n % BLOCK_SIZE < BLOCK_Y) { // only run once for every block_x iterations
      __syncthreads();
      int m = tIdx + (n / BLOCK_SIZE) * BLOCK_SIZE;
      if (m < totalpairs) {
        int jj = pairidx_to_j(m);
        int kk = m - jj * (jj - 1) / 2; // 0-indexed
        jj = jnum - jj - 1;
        kk += jj + 1;
        const DataT Rij = sdist[jj];
        const DataT Rik = sdist[kk];
        s_theta[tIdx] =
            acosf(0.95 * (svec[jj].x * svec[kk].x + svec[jj].y * svec[kk].y + svec[jj].z * svec[kk].z) / (Rij * Rik));
      }
      __syncthreads();
    }
    // run angular calculation
    if (n < totalpairs) {
      int jj = pairidx_to_j(n);
      int kk = n - jj * (jj - 1) / 2; // 0-indexed
      jj = jnum - jj - 1;
      kk += jj + 1;
      // printf("n %d, jnum %d, jj %d, kk %d\n", n, jnum, jj, kk);
      const DataT Rij = sdist[jj];
      SpeciesT type_j = stype[jj];
      DataT fc_ij = sfc[jj];
      const DataT Rik = sdist[kk];
      SpeciesT type_k = stype[kk];
      DataT fc_ik = sfc[kk];

      DataT theta = s_theta[n % BLOCK_SIZE];
      // DataT theta = 1.0;
      DataT Rijk = (Rij + Rik) / 2;
      DataT fc_ijk = fc_ij * fc_ik;

      IndexT subaev_offset = angular_sublength * csubaev_offsets(type_j, type_k, num_species);

      for (int itheta = tile.x; itheta < nShfZ; itheta += TILEX) {
        DataT ShfZ = __ldg(&ShfZ_t[itheta]);

        DataT factor1 = __powf((1 + __cosf(theta - ShfZ)) / 2, Zeta);
        // DataT factor1 = 1.0;

        for (int ishfr = tile.y; ishfr < nShfA; ishfr += TILEY) {
          DataT ShfA = __ldg(&ShfA_t[ishfr]);
          DataT factor2 = __expf(-EtaA * (Rijk - ShfA) * (Rijk - ShfA));
          // DataT factor2 = 1.0;

          DataT res = 2 * factor1 * factor2 * fc_ijk;

          atomicAdd(&saev[subaev_offset + ishfr * nShfZ + itheta], res);
          // saev[subaev_offset + ishfr * nShfZ + itheta] += res;
          // saev[0] = res;
        }
      }
    }
  }
  __syncthreads();

  for (int iaev = tIdx; iaev < angular_length; iaev += blockDim.x * blockDim.y) {
    aev_t[mol_idx][i][radial_length + iaev] = saev[iaev];
  }
}

template <
    bool is_double_backward,
    typename SpeciesT,
    typename DataT,
    typename IndexT = int,
    int TILEX = 8,
    int TILEY = 4>
__global__ void cuAngularAEVs_backward_or_doublebackward(
    torch::PackedTensorAccessor32<SpeciesT, 2, torch::RestrictPtrTraits> species_t,
    torch::PackedTensorAccessor32<DataT, 3, torch::RestrictPtrTraits> pos_t,
    torch::PackedTensorAccessor32<DataT, 1, torch::RestrictPtrTraits> ShfA_t,
    torch::PackedTensorAccessor32<DataT, 1, torch::RestrictPtrTraits> ShfZ_t,
    torch::PackedTensorAccessor32<DataT, 1, torch::RestrictPtrTraits> EtaA_t,
    torch::PackedTensorAccessor32<DataT, 1, torch::RestrictPtrTraits> Zeta_t,
    torch::PackedTensorAccessor32<DataT, 3, torch::RestrictPtrTraits>
        grad_output, // for backward, this is daev, for double backward, this is dforce (i.e. ddcoord)
    torch::PackedTensorAccessor32<DataT, 3, torch::RestrictPtrTraits>
        grad_input, // for backward, this is dcoord, for double backward, this is ddaev
    const torch::PackedTensorAccessor32<SpeciesT, 1, torch::RestrictPtrTraits> atomJ,
    const torch::PackedTensorAccessor32<DataT, 1, torch::RestrictPtrTraits> distJ,
    const AtomI* __restrict__ atom_i,
    int* d_nPairsPerCenterAtom,
    int* d_centerAtomStartIdx,
    float Rca,
    int angular_length,
    int angular_sublength,
    int radial_length,
    int num_species,
    int maxnbrs_per_atom_aligned,
    int angular_length_aligned,
    int ncentral_atoms) {
  extern __shared__ DataT smem[];

  constexpr int threads_per_catom = TILEX * TILEY;
  static_assert(threads_per_catom == C10_WARP_SIZE);
  int gIdx = blockIdx.x * blockDim.x + threadIdx.x;
  int cIdx = gIdx / threads_per_catom; // central atom id
  int jnum = d_nPairsPerCenterAtom[cIdx];

  if (cIdx >= ncentral_atoms)
    return;
  if (jnum < 2)
    return;

  int groupIdx = threadIdx.x / threads_per_catom;
  int laneIdx = threadIdx.x % threads_per_catom;
  int ncatom_per_tpb = blockDim.x / threads_per_catom; // e.g. 2 catom per block

  DataT* sdx = &smem[groupIdx * maxnbrs_per_atom_aligned];
  int offset = ncatom_per_tpb * maxnbrs_per_atom_aligned;

  DataT* sdy = &smem[offset + groupIdx * maxnbrs_per_atom_aligned];
  offset += ncatom_per_tpb * maxnbrs_per_atom_aligned;

  DataT* sdz = &smem[offset + groupIdx * maxnbrs_per_atom_aligned];
  offset += ncatom_per_tpb * maxnbrs_per_atom_aligned;

  DataT* sdjx_grad = &smem[offset + groupIdx * maxnbrs_per_atom_aligned];
  offset += ncatom_per_tpb * maxnbrs_per_atom_aligned;

  DataT* sdjy_grad = &smem[offset + groupIdx * maxnbrs_per_atom_aligned];
  offset += ncatom_per_tpb * maxnbrs_per_atom_aligned;

  DataT* sdjz_grad = &smem[offset + groupIdx * maxnbrs_per_atom_aligned];
  offset += ncatom_per_tpb * maxnbrs_per_atom_aligned;

  DataT* sdist = &smem[offset + groupIdx * maxnbrs_per_atom_aligned];
  offset += ncatom_per_tpb * maxnbrs_per_atom_aligned;

  DataT* sfc = &smem[offset + groupIdx * maxnbrs_per_atom_aligned];
  offset += ncatom_per_tpb * maxnbrs_per_atom_aligned;

  DataT* sfc_grad = &smem[offset + groupIdx * maxnbrs_per_atom_aligned];
  offset += ncatom_per_tpb * maxnbrs_per_atom_aligned;

  int* stype = (int*)&smem[offset + groupIdx * maxnbrs_per_atom_aligned];

  DataT EtaA = EtaA_t[0];
  DataT Zeta = Zeta_t[0];

  IndexT nShfA = ShfA_t.size(0);
  IndexT nShfZ = ShfZ_t.size(0);

  int start_idx = d_centerAtomStartIdx[cIdx];
  AtomI aI = atom_i[cIdx];
  int mol_idx = aI.midx;
  int i = aI.i;

  DataT xi = pos_t[mol_idx][i][0];
  DataT yi = pos_t[mol_idx][i][1];
  DataT zi = pos_t[mol_idx][i][2];

  for (int jj = laneIdx; jj < jnum; jj += threads_per_catom) {
    DataT Rij = distJ[start_idx + jj];
    int j = atomJ[start_idx + jj];
    SpeciesT type_j = species_t[mol_idx][j];
    sdx[jj] = pos_t[mol_idx][j][0] - xi;
    sdy[jj] = pos_t[mol_idx][j][1] - yi;
    sdz[jj] = pos_t[mol_idx][j][2] - zi;
    stype[jj] = type_j;
    sdist[jj] = Rij;
    // cutoff
    DataT fc_ij = 0.5 * cos(PI * Rij / Rca) + 0.5;
    DataT fc_ij_grad = -0.5 * (PI / Rca) * sin(PI * Rij / Rca);
    sfc[jj] = fc_ij;
    sfc_grad[jj] = fc_ij_grad;
  }

  // grad init
  DataT sdix_grad = 0;
  DataT sdiy_grad = 0;
  DataT sdiz_grad = 0;

  for (int jj = laneIdx; jj < jnum; jj += threads_per_catom) {
    sdjx_grad[jj] = 0;
    sdjy_grad[jj] = 0;
    sdjz_grad[jj] = 0;
  }

  short2 tile = make_short2(laneIdx % TILEX, laneIdx / TILEX);
  const DataT tc = 0.95; // theta constant factor
  // must sync if threads_per_catom != 32 (wrap size) to make sure shared data is ready
  // __syncthreads

  for (int jj = 0; jj < jnum; jj++) {
    const DataT Rij = sdist[jj];
    SpeciesT type_j = stype[jj];

    DataT fc_ij = sfc[jj];
    DataT grad_fc_ij = sfc_grad[jj];

    for (int kk_start = jj + 1; kk_start < jnum; kk_start += threads_per_catom) {
      int kk = kk_start + laneIdx;
      DataT theta = 0;
      DataT grad_theta_vij_x = 0;
      DataT grad_theta_vij_y = 0;
      DataT grad_theta_vij_z = 0;
      DataT grad_theta_vik_x = 0;
      DataT grad_theta_vik_y = 0;
      DataT grad_theta_vik_z = 0;
      if (kk < jnum) {
        const DataT Rik = sdist[kk];
        DataT vij_vik_dot = sdx[jj] * sdx[kk] + sdy[jj] * sdy[kk] + sdz[jj] * sdz[kk];
        theta = acos(tc * vij_vik_dot / (Rij * Rik));
        // grad
        DataT vij_factor =
            tc / (Rij * Rij * Rij * sqrt(-tc * tc * vij_vik_dot * vij_vik_dot / (Rij * Rij) + Rik * Rik));
        DataT vik_factor = tc /
            (Rik * Rik * Rik *
             sqrt(-tc * tc * vij_vik_dot * vij_vik_dot / (Rik * Rik) + Rij * Rij)); // tricky 80ms improved
        grad_theta_vij_x = vij_factor * (sdx[jj] * vij_vik_dot - sdx[kk] * Rij * Rij);
        grad_theta_vij_y = vij_factor * (sdy[jj] * vij_vik_dot - sdy[kk] * Rij * Rij);
        grad_theta_vij_z = vij_factor * (sdz[jj] * vij_vik_dot - sdz[kk] * Rij * Rij);
        grad_theta_vik_x = vik_factor * (sdx[kk] * vij_vik_dot - sdx[jj] * Rik * Rik);
        grad_theta_vik_y = vik_factor * (sdy[kk] * vij_vik_dot - sdy[jj] * Rik * Rik);
        grad_theta_vik_z = vik_factor * (sdz[kk] * vij_vik_dot - sdz[jj] * Rik * Rik);
      }

      for (int srcLane = 0; srcLane < C10_WARP_SIZE && (kk_start + srcLane) < jnum; ++srcLane) {
        int kk = kk_start + srcLane;
        DataT theta_ijk = __shfl_sync(0xFFFFFFFF, theta, srcLane);
        // TODO necessary?
        DataT grad_theta_vij_x_ = __shfl_sync(0xFFFFFFFF, grad_theta_vij_x, srcLane);
        DataT grad_theta_vij_y_ = __shfl_sync(0xFFFFFFFF, grad_theta_vij_y, srcLane);
        DataT grad_theta_vij_z_ = __shfl_sync(0xFFFFFFFF, grad_theta_vij_z, srcLane);
        DataT grad_theta_vik_x_ = __shfl_sync(0xFFFFFFFF, grad_theta_vik_x, srcLane);
        DataT grad_theta_vik_y_ = __shfl_sync(0xFFFFFFFF, grad_theta_vik_y, srcLane);
        DataT grad_theta_vik_z_ = __shfl_sync(0xFFFFFFFF, grad_theta_vik_z, srcLane);

        const DataT Rik = sdist[kk];
        SpeciesT type_k = stype[kk];

        DataT fc_ik = sfc[kk];
        DataT grad_fc_ik = sfc_grad[kk];

        DataT Rijk = (Rij + Rik) / 2;
        DataT fc_ijk = fc_ij * fc_ik;

        IndexT subaev_offset = angular_sublength * csubaev_offsets(type_j, type_k, num_species);

        for (int itheta = tile.x; itheta < nShfZ; itheta += TILEX) {
          DataT ShfZ = __ldg(&ShfZ_t[itheta]);

          DataT factor1 = pow((1 + cos(theta_ijk - ShfZ)) / 2, Zeta);
          DataT grad_factor1_theta = 1.0 / 2.0 * Zeta * pow((1 + cos(ShfZ - theta_ijk)) / 2, Zeta - 1) *
              sin(ShfZ - theta_ijk); // tricky 100ms improved

          for (int ishfr = tile.y; ishfr < nShfA; ishfr += TILEY) {
            DataT ShfA = __ldg(&ShfA_t[ishfr]);
            DataT factor2 = exp(-EtaA * (Rijk - ShfA) * (Rijk - ShfA));
            DataT grad_factor2_dist = -EtaA * (Rijk - ShfA) * factor2;

            DataT grad_vij_x = 2 *
                (grad_factor1_theta * grad_theta_vij_x_ * factor2 * fc_ijk +
                 factor1 * grad_factor2_dist * sdx[jj] / Rij * fc_ijk +
                 factor1 * factor2 * fc_ik * grad_fc_ij * sdx[jj] / Rij);
            DataT grad_vij_y = 2 *
                (grad_factor1_theta * grad_theta_vij_y_ * factor2 * fc_ijk +
                 factor1 * grad_factor2_dist * sdy[jj] / Rij * fc_ijk +
                 factor1 * factor2 * fc_ik * grad_fc_ij * sdy[jj] / Rij);
            DataT grad_vij_z = 2 *
                (grad_factor1_theta * grad_theta_vij_z_ * factor2 * fc_ijk +
                 factor1 * grad_factor2_dist * sdz[jj] / Rij * fc_ijk +
                 factor1 * factor2 * fc_ik * grad_fc_ij * sdz[jj] / Rij);
            DataT grad_vik_x = 2 *
                (grad_factor1_theta * grad_theta_vik_x_ * factor2 * fc_ijk +
                 factor1 * grad_factor2_dist * sdx[kk] / Rik * fc_ijk +
                 factor1 * factor2 * fc_ij * grad_fc_ik * sdx[kk] / Rik);
            DataT grad_vik_y = 2 *
                (grad_factor1_theta * grad_theta_vik_y_ * factor2 * fc_ijk +
                 factor1 * grad_factor2_dist * sdy[kk] / Rik * fc_ijk +
                 factor1 * factor2 * fc_ij * grad_fc_ik * sdy[kk] / Rik);
            DataT grad_vik_z = 2 *
                (grad_factor1_theta * grad_theta_vik_z_ * factor2 * fc_ijk +
                 factor1 * grad_factor2_dist * sdz[kk] / Rik * fc_ijk +
                 factor1 * factor2 * fc_ij * grad_fc_ik * sdz[kk] / Rik);

            if (is_double_backward) {
              int atomj_idx = atomJ[start_idx + jj];
              int atomk_idx = atomJ[start_idx + kk];
              auto& grad_force = grad_output;
              auto& grad_grad_aev = grad_input;
              grad_vij_x *= (grad_force[mol_idx][atomj_idx][0] - grad_force[mol_idx][i][0]);
              grad_vij_y *= (grad_force[mol_idx][atomj_idx][1] - grad_force[mol_idx][i][1]);
              grad_vij_z *= (grad_force[mol_idx][atomj_idx][2] - grad_force[mol_idx][i][2]);
              grad_vik_x *= (grad_force[mol_idx][atomk_idx][0] - grad_force[mol_idx][i][0]);
              grad_vik_y *= (grad_force[mol_idx][atomk_idx][1] - grad_force[mol_idx][i][1]);
              grad_vik_z *= (grad_force[mol_idx][atomk_idx][2] - grad_force[mol_idx][i][2]);
              atomicAdd(
                  &grad_grad_aev[mol_idx][i][radial_length + subaev_offset + ishfr * nShfZ + itheta],
                  grad_vij_x + grad_vij_y + grad_vij_z + grad_vik_x + grad_vik_y + grad_vik_z);
            } else {
              DataT grad_output_item = grad_output[mol_idx][i][radial_length + subaev_offset + ishfr * nShfZ + itheta];
              grad_vij_x *= grad_output_item;
              grad_vij_y *= grad_output_item;
              grad_vij_z *= grad_output_item;
              grad_vik_x *= grad_output_item;
              grad_vik_y *= grad_output_item;
              grad_vik_z *= grad_output_item;

              sdix_grad += (-grad_vij_x - grad_vik_x);
              sdiy_grad += (-grad_vij_y - grad_vik_y);
              sdiz_grad += (-grad_vij_z - grad_vik_z);

              for (int offset = 16; offset > 0; offset /= 2) {
                grad_vij_x += __shfl_down_sync(0xFFFFFFFF, grad_vij_x, offset);
                grad_vij_y += __shfl_down_sync(0xFFFFFFFF, grad_vij_y, offset);
                grad_vij_z += __shfl_down_sync(0xFFFFFFFF, grad_vij_z, offset);
                grad_vik_x += __shfl_down_sync(0xFFFFFFFF, grad_vik_x, offset);
                grad_vik_y += __shfl_down_sync(0xFFFFFFFF, grad_vik_y, offset);
                grad_vik_z += __shfl_down_sync(0xFFFFFFFF, grad_vik_z, offset);
              }
              if (laneIdx == 0) {
                sdjx_grad[jj] += grad_vij_x;
                sdjy_grad[jj] += grad_vij_y;
                sdjz_grad[jj] += grad_vij_z;

                sdjx_grad[kk] += grad_vik_x;
                sdjy_grad[kk] += grad_vik_y;
                sdjz_grad[kk] += grad_vik_z;
              }
            }
          }
        }
      }
    }
  }

  if (!is_double_backward) {
    auto& grad_coord = grad_input;
    int atomi_idx = i;
    atomicAdd(&grad_coord[mol_idx][atomi_idx][0], sdix_grad);
    atomicAdd(&grad_coord[mol_idx][atomi_idx][1], sdiy_grad);
    atomicAdd(&grad_coord[mol_idx][atomi_idx][2], sdiz_grad);

    for (int jj = laneIdx; jj < jnum; jj += threads_per_catom) {
      int atomj_idx = atomJ[start_idx + jj];

      atomicAdd(&grad_coord[mol_idx][atomj_idx][0], sdjx_grad[jj]);
      atomicAdd(&grad_coord[mol_idx][atomj_idx][1], sdjy_grad[jj]);
      atomicAdd(&grad_coord[mol_idx][atomj_idx][2], sdjz_grad[jj]);
    }
  }
}

template <typename SpeciesT, typename DataT>
__global__ void cuRadialAEVs(
    torch::PackedTensorAccessor32<SpeciesT, 2, torch::RestrictPtrTraits> species_t,
    torch::PackedTensorAccessor32<DataT, 1, torch::RestrictPtrTraits> ShfR_t,
    torch::PackedTensorAccessor32<DataT, 1, torch::RestrictPtrTraits> EtaR_t,
    torch::PackedTensorAccessor32<DataT, 3, torch::RestrictPtrTraits> aev_t,
    const int* __restrict__ atomJ,
    const float* __restrict__ distJ,
    const AtomI* __restrict__ atom_i,
    const int* __restrict__ d_nPairsPerCenterAtom,
    const int* __restrict__ d_centerAtomStartIdx,
    float Rcr,
    int radial_length,
    int radial_sublength,
    int nRadialRij,
    int max_numPairsPerAtom) {
  extern __shared__ DataT smem[];

  DataT* s_radial = &smem[0];
  DataT* s_fc = &smem[radial_length];

  int cIdx = blockIdx.x; // central atom id
  int tIdx = threadIdx.y * blockDim.x + threadIdx.x; // local thread idx

  int nShfR = ShfR_t.size(0);
  DataT EtaR = EtaR_t[0];

  int start_idx = d_centerAtomStartIdx[cIdx];
  int jnum = d_nPairsPerCenterAtom[cIdx];

  if (jnum < 1)
    return;

  AtomI aI = atom_i[cIdx];
  int mol_idx = aI.midx;
  int i = aI.i;
  int laneIdx = threadIdx.x % blockDim.x;

  for (int iaev = tIdx; iaev < radial_length; iaev += blockDim.x * blockDim.y) {
    s_radial[iaev] = 0;
  }

  // TODO move fc outside of radial kernel, so could directly load
  for (int jj = tIdx; jj < jnum; jj += blockDim.x * blockDim.y) {
    DataT Rij = distJ[start_idx + jj];
    s_fc[jj] = 0.5 * __cosf(PI * Rij / Rcr) + 0.5;
  }
  __syncthreads();

  for (int jj = threadIdx.y; jj < jnum; jj += blockDim.y) {
    DataT fc = s_fc[jj];
    DataT Rij = distJ[start_idx + jj];
    int j = atomJ[start_idx + jj];
    SpeciesT type_j = species_t[mol_idx][j];

    for (int ishfr = laneIdx; ishfr < nShfR; ishfr += blockDim.x) {
      DataT ShfR = __ldg(&ShfR_t[ishfr]);
      DataT GmR = 0.25 * __expf(-EtaR * (Rij - ShfR) * (Rij - ShfR)) * fc;
      atomicAdd(&s_radial[type_j * radial_sublength + ishfr], GmR);
    }
  }

  __syncthreads();
  for (int iaev = tIdx; iaev < radial_length; iaev += blockDim.x * blockDim.y) {
    aev_t[mol_idx][i][iaev] = s_radial[iaev];
  }
}

// every <THREADS_PER_RIJ> threads take care of 1 RIJ, and iterate <nShfR / THREADS_PER_RIJ> times
template <bool is_double_backward, typename SpeciesT, typename DataT, int THREADS_PER_RIJ>
__global__ void cuRadialAEVs_backward_or_doublebackward(
    const torch::PackedTensorAccessor32<DataT, 3, torch::RestrictPtrTraits> pos_t,
    const torch::PackedTensorAccessor32<SpeciesT, 2, torch::RestrictPtrTraits> species_t,
    const torch::PackedTensorAccessor32<DataT, 1, torch::RestrictPtrTraits> ShfR_t,
    const torch::PackedTensorAccessor32<DataT, 1, torch::RestrictPtrTraits> EtaR_t,
    torch::PackedTensorAccessor32<DataT, 3, torch::RestrictPtrTraits>
        grad_aev, // daev for backward, ddaev for double backward
    torch::PackedTensorAccessor32<DataT, 1, torch::RestrictPtrTraits>
        grad_dist, // ddist for backward, dddist for double backward
    torch::PackedTensorAccessor32<DataT, 3, torch::RestrictPtrTraits>
        grad_coord_or_force, // dcoord for backward, dforce(i.e. ddcoord) for double backward
    const torch::PackedTensorAccessor32<SpeciesT, 1, torch::RestrictPtrTraits> atomJ,
    const torch::PackedTensorAccessor32<DataT, 1, torch::RestrictPtrTraits> distJ,
    const AtomI* __restrict__ atom_i,
    const int* __restrict__ d_nPairsPerCenterAtom,
    const int* __restrict__ d_centerAtomStartIdx,
    float Rcr,
    int radial_length,
    int radial_sublength,
    int nRadialRij,
    int max_numPairsPerAtom) {
  extern __shared__ DataT smem[];
  DataT* s_grad_dist = &smem[0]; // ddist for backward, dddist for double backward
  DataT* s_dcoord_or_ddaev = &smem[max_numPairsPerAtom];
  // double backward
  auto& s_ddaev = s_dcoord_or_ddaev;
  // backward
  float3* s_dcoord_j = reinterpret_cast<float3*>(s_dcoord_or_ddaev);
  __shared__ float3 s_dcoord_i;
  const float3* pos_t_3 = reinterpret_cast<const float3*>(&pos_t[0][0][0]);
  const int max_natoms_per_mol = pos_t.size(1);

  int cIdx = blockIdx.x; // central atom id
  int tIdx = threadIdx.y * blockDim.x + threadIdx.x; // local thread idx

  int nShfR = ShfR_t.size(0);
  DataT EtaR = EtaR_t[0];

  int start_idx = d_centerAtomStartIdx[cIdx];
  int jnum = d_nPairsPerCenterAtom[cIdx];

  if (jnum < 1)
    return;

  AtomI aI = atom_i[cIdx];
  int mol_idx = aI.midx;
  int i = aI.i;
  int laneIdx = threadIdx.x % blockDim.x;

  float3 pos_i = pos_t_3[mol_idx * max_natoms_per_mol + i];

  if (is_double_backward) {
    float3* grad_force = reinterpret_cast<float3*>(&grad_coord_or_force[0][0][0]);
    float3 dforce_i = grad_force[mol_idx * max_natoms_per_mol + i];
    for (int jj = tIdx; jj < jnum; jj += blockDim.x * blockDim.y) {
      DataT Rij = distJ[start_idx + jj];
      int j = atomJ[start_idx + jj];
      float3 pos_j = pos_t_3[mol_idx * max_natoms_per_mol + j];
      float3 dforce_j = grad_force[mol_idx * max_natoms_per_mol + j];
      float3 delta = make_float3(pos_j.x - pos_i.x, pos_j.y - pos_i.y, pos_j.z - pos_i.z);
      s_grad_dist[jj] = ((dforce_j.x - dforce_i.x) * delta.x + (dforce_j.y - dforce_i.y) * delta.y +
                         (dforce_j.z - dforce_i.z) * delta.z) /
          Rij;
    }
    for (int iaev = tIdx; iaev < radial_length; iaev += blockDim.x * blockDim.y) {
      s_ddaev[iaev] = 0;
    }
  } else {
    for (int jj = tIdx; jj < jnum; jj += blockDim.x * blockDim.y) {
      s_grad_dist[jj] = 0;
    }
    for (int jj = tIdx; jj < jnum; jj += blockDim.x * blockDim.y) {
      s_dcoord_j[jj] = make_float3(0.f, 0.f, 0.f);
    }
  }
  if (threadIdx.x == 0)
    s_dcoord_i = make_float3(0.f, 0.f, 0.f);

  __syncthreads();

  DataT upstream_grad;
  for (int jj = threadIdx.y; jj < jnum; jj += blockDim.y) {
    DataT Rij = distJ[start_idx + jj];
    int j = atomJ[start_idx + jj];
    // DataT fc = 0.5;
    // DataT fc_grad = -0.5;
    DataT fc = 0.5 * __cosf(PI * Rij / Rcr) + 0.5;
    DataT fc_grad = -0.5 * (PI / Rcr) * __sinf(PI * Rij / Rcr);
    SpeciesT type_j = species_t[mol_idx][j];

    if (is_double_backward) {
      upstream_grad = s_grad_dist[jj];
    }

    for (int ishfr = laneIdx; ishfr < nShfR; ishfr += blockDim.x) {
      DataT ShfR = __ldg(&ShfR_t[ishfr]);

      DataT GmR = 0.25 * __expf(-EtaR * (Rij - ShfR) * (Rij - ShfR));
      DataT GmR_grad = -EtaR * (-2 * ShfR + 2 * Rij) * GmR;
      DataT jacobian = GmR_grad * fc + GmR * fc_grad;

      if (is_double_backward) {
        auto& grad_grad_aev = grad_aev;
        atomicAdd(&grad_grad_aev[mol_idx][i][type_j * radial_sublength + ishfr], upstream_grad * jacobian);
      } else {
        upstream_grad = grad_aev[mol_idx][i][type_j * radial_sublength + ishfr];
        atomicAdd(&s_grad_dist[jj], upstream_grad * jacobian);
      }
    }
  }

  if (!is_double_backward) {
    __syncthreads();
    auto& grad_coord = grad_coord_or_force;
    for (int jj = tIdx; jj < jnum; jj += blockDim.x * blockDim.y) {
      DataT Rij = distJ[start_idx + jj];
      int j = atomJ[start_idx + jj];
      float3 pos_j = pos_t_3[mol_idx * max_natoms_per_mol + j];
      float3 delta = make_float3(pos_j.x - pos_i.x, pos_j.y - pos_i.y, pos_j.z - pos_i.z);

      DataT grad_dist_coord_x = delta.x / Rij;
      DataT grad_dist_coord_y = delta.y / Rij;
      DataT grad_dist_coord_z = delta.z / Rij;
      DataT grad_radial_dist_item = s_grad_dist[jj];

      atomicAdd(&grad_coord[mol_idx][j][0], grad_radial_dist_item * grad_dist_coord_x);
      atomicAdd(&grad_coord[mol_idx][j][1], grad_radial_dist_item * grad_dist_coord_y);
      atomicAdd(&grad_coord[mol_idx][j][2], grad_radial_dist_item * grad_dist_coord_z);

      atomicAdd(&grad_coord[mol_idx][i][0], -grad_radial_dist_item * grad_dist_coord_x);
      atomicAdd(&grad_coord[mol_idx][i][1], -grad_radial_dist_item * grad_dist_coord_y);
      atomicAdd(&grad_coord[mol_idx][i][2], -grad_radial_dist_item * grad_dist_coord_z);

      // atomicAdd(&s_dcoord_i.x, -grad_radial_dist_item * grad_dist_coord_x);
      // atomicAdd(&s_dcoord_i.y, -grad_radial_dist_item * grad_dist_coord_y);
      // atomicAdd(&s_dcoord_i.z, -grad_radial_dist_item * grad_dist_coord_z);
    }
    __syncthreads();

    // if (tIdx == 0) {
    //   atomicAdd(&grad_coord[mol_idx][i][0], s_dcoord_i.x);
    //   atomicAdd(&grad_coord[mol_idx][i][1], s_dcoord_i.y);
    //   atomicAdd(&grad_coord[mol_idx][i][2], s_dcoord_i.z);
    // }
  }
}

template <int ATOM_I_PER_BLOCK>
__global__ void cutoffSelect(
    int* __restrict__ atomJ,
    float* __restrict__ distJ,
    const AtomI* __restrict__ atom_i,
    int* __restrict__ radial_atomJ,
    float* __restrict__ radial_distJ,
    int* __restrict__ angular_atomJ,
    float* __restrict__ angular_distJ,
    const int* __restrict__ nums_per_row,
    const int* __restrict__ startidx_per_row,
    float new_cutoff,
    int* __restrict__ new_nums_per_row,
    int num_rows,
    int max_natoms_per_mol) {
  __shared__ int s_new_pcounter_i[ATOM_I_PER_BLOCK];
  __shared__ int s_num_max;
  int gi = blockIdx.x * blockDim.y + threadIdx.y;
  if (gi >= num_rows)
    return;

  AtomI aI = atom_i[gi];
  int i = aI.i;
  int mol_idx = aI.midx;
  int jnum = nums_per_row[gi];
  int start_i = startidx_per_row[gi];
  int ii = threadIdx.y;
  int idx = blockDim.x * threadIdx.y + threadIdx.x;
  int natom_pairs = max_natoms_per_mol * (max_natoms_per_mol - 1);

  if (idx < ATOM_I_PER_BLOCK) {
    s_new_pcounter_i[idx] = 0;
    int ii = blockIdx.x * blockDim.y + idx;
    int num_max = ii < num_rows ? nums_per_row[ii] : 0;

    for (int offset = 16; offset > 0; offset /= 2) {
      num_max = max(num_max, __shfl_down_sync(0xFFFFFFFF, num_max, offset));
    }
    if (idx == 0) {
      s_num_max = num_max;
    }
  }
  __syncthreads();

  for (int jj = threadIdx.x; jj < s_num_max && jj < jnum; jj += blockDim.x) {
    // printf("cutoff1, mol: %d, Rij: %f\n", mol_idx, d.Rij);
    int j = atomJ[natom_pairs * mol_idx + i * (max_natoms_per_mol - 1) + jj];
    float dist = distJ[natom_pairs * mol_idx + i * (max_natoms_per_mol - 1) + jj];
    radial_atomJ[start_i + jj] = j;
    radial_distJ[start_i + jj] = dist;
    if (dist <= new_cutoff) {
      int pidx = atomicAdd(&s_new_pcounter_i[ii], 1);
      angular_atomJ[start_i + pidx] = j;
      angular_distJ[start_i + pidx] = dist;
    }
  }
  __syncthreads();

  gi = idx + blockIdx.x * blockDim.y;
  if (idx < blockDim.y && gi < num_rows) {
    new_nums_per_row[gi] = s_new_pcounter_i[idx];
  }
}

template <typename DataT>
void cubScan(const DataT* d_in, DataT* d_out, int num_items, cudaStream_t stream) {
  auto& allocator = *c10::cuda::CUDACachingAllocator::get();

  // Determine temporary device storage requirements
  void* d_temp_storage = NULL;
  size_t temp_storage_bytes = 0;
  cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, d_in, d_out, num_items, stream);

  // Allocate temporary storage
  auto buffer_tmp = allocator.allocate(temp_storage_bytes);
  d_temp_storage = buffer_tmp.get();

  // Run exclusive prefix sum
  cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, d_in, d_out, num_items, stream);
}

template <typename DataT, typename LambdaOpT>
int cubDeviceSelectIf(const DataT* d_in, DataT* d_out, int num_items, LambdaOpT select_op, cudaStream_t stream) {
  auto& allocator = *c10::cuda::CUDACachingAllocator::get();
  auto buffer_count = allocator.allocate(sizeof(int));
  int* d_num_selected_out = (int*)buffer_count.get();

  // Determine temporary device storage requirements
  void* d_temp_storage = NULL;
  size_t temp_storage_bytes = 0;
  cub::DeviceSelect::If(
      d_temp_storage, temp_storage_bytes, d_in, d_out, d_num_selected_out, num_items, select_op, stream);

  // Allocate temporary storage
  auto buffer_tmp = allocator.allocate(temp_storage_bytes);
  d_temp_storage = buffer_tmp.get();

  // Run selection
  cub::DeviceSelect::If(
      d_temp_storage, temp_storage_bytes, d_in, d_out, d_num_selected_out, num_items, select_op, stream);

  int num_selected = 0;
  cudaMemcpyAsync(&num_selected, d_num_selected_out, sizeof(int), cudaMemcpyDefault, stream);
  cudaStreamSynchronize(stream);

  return num_selected;
}

template <typename DataT>
int cubDeviceSelectFlagged(const DataT* d_in, DataT* d_out, int num_items, char* d_flags, cudaStream_t stream) {
  auto& allocator = *c10::cuda::CUDACachingAllocator::get();
  auto buffer_count = allocator.allocate(sizeof(int));
  int* d_num_selected_out = (int*)buffer_count.get();

  // Determine temporary device storage requirements
  void* d_temp_storage = NULL;
  size_t temp_storage_bytes = 0;
  cub::DeviceSelect::Flagged(
      d_temp_storage, temp_storage_bytes, d_in, d_flags, d_out, d_num_selected_out, num_items, stream);

  // Allocate temporary storage
  auto buffer_tmp = allocator.allocate(temp_storage_bytes);
  d_temp_storage = buffer_tmp.get();

  // Run selection
  cub::DeviceSelect::Flagged(
      d_temp_storage, temp_storage_bytes, d_in, d_flags, d_out, d_num_selected_out, num_items, stream);

  int num_selected = 0;
  cudaMemcpyAsync(&num_selected, d_num_selected_out, sizeof(int), cudaMemcpyDefault, stream);
  cudaStreamSynchronize(stream);

  return num_selected;
}

template <typename DataT>
DataT cubMax(const DataT* d_in, int num_items, cudaStream_t stream) {
  auto& allocator = *c10::cuda::CUDACachingAllocator::get();
  auto buffer_count = allocator.allocate(sizeof(DataT));
  DataT* d_out = (DataT*)buffer_count.get();

  // Determine temporary device storage requirements
  void* d_temp_storage = NULL;
  size_t temp_storage_bytes = 0;
  cub::DeviceReduce::Max(d_temp_storage, temp_storage_bytes, d_in, d_out, num_items, stream);

  // Allocate temporary storage
  auto buffer_tmp = allocator.allocate(temp_storage_bytes);
  d_temp_storage = buffer_tmp.get();

  // Run min-reduction
  cub::DeviceReduce::Max(d_temp_storage, temp_storage_bytes, d_in, d_out, num_items, stream);

  DataT maxVal = 0;
  cudaMemcpyAsync(&maxVal, d_out, sizeof(DataT), cudaMemcpyDefault, stream);
  cudaStreamSynchronize(stream);

  return maxVal;
}

template <typename DataT>
DataT cubSum(const DataT* d_in, int num_items, cudaStream_t stream) {
  auto& allocator = *c10::cuda::CUDACachingAllocator::get();
  auto buffer_count = allocator.allocate(sizeof(DataT));
  DataT* d_out = (DataT*)buffer_count.get();

  // Determine temporary device storage requirements
  void* d_temp_storage = NULL;
  size_t temp_storage_bytes = 0;
  cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, d_in, d_out, num_items, stream);

  // Allocate temporary storage
  auto buffer_tmp = allocator.allocate(temp_storage_bytes);
  d_temp_storage = buffer_tmp.get();

  // Run sum-reduction
  cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, d_in, d_out, num_items, stream);

  DataT sumVal = 0;
  cudaMemcpyAsync(&sumVal, d_out, sizeof(DataT), cudaMemcpyDefault, stream);
  cudaStreamSynchronize(stream);

  return sumVal;
}

// NOTE: assumes size of EtaA_t = Zeta_t = EtaR_t = 1
void cuaev_forward(
    const Tensor& coordinates_t,
    const Tensor& species_t,
    const AEVScalarParams& aev_params,
    Result& result) {
  TORCH_CHECK(
      (species_t.dtype() == torch::kInt32) && (coordinates_t.dtype() == torch::kFloat32), "Unsupported input type");
  TORCH_CHECK(
      aev_params.EtaR_t.size(0) == 1 || aev_params.EtaA_t.size(0) == 1 || aev_params.Zeta_t.size(0) == 1,
      "cuda extension is currently not supported for the specified "
      "configuration");

  float Rcr = aev_params.Rcr;
  float Rca = aev_params.Rca;
  const int n_molecules = species_t.size(0);
  const int max_natoms_per_mol = species_t.size(1);
  int aev_length = aev_params.radial_length + aev_params.angular_length;
  int total_atoms = n_molecules * max_natoms_per_mol;

  result.aev_t = torch::zeros({n_molecules, max_natoms_per_mol, aev_length}, coordinates_t.options());
  // TODO replace zeros with empty, need padding atom also run kernel
  // result.aev_t = torch::empty({n_molecules, max_natoms_per_mol, aev_length}, coordinates_t.options());
  if (species_t.numel() == 0) {
    return;
  }

  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  auto thrust_allocator = THCThrustAllocator(at::globalContext().lazyInitCUDA());
  auto policy = thrust::cuda::par(thrust_allocator).on(stream);
  auto& allocator = *c10::cuda::CUDACachingAllocator::get();

  // buffer to store all the pairwise distance (Rij)
  int pairs_per_mol = max_natoms_per_mol * (max_natoms_per_mol - 1);
  auto total_natom_pairs = n_molecules * pairs_per_mol;
  auto d_options = torch::dtype(torch::kUInt8).device(coordinates_t.device());

  Tensor atomJ_t = torch::empty(total_natom_pairs, d_options.dtype(torch::kInt32));
  int* atomJ_p = (int*)atomJ_t.data_ptr();
  Tensor distJ_t = torch::empty(total_natom_pairs, d_options.dtype(torch::kFloat32));
  float* distJ_p = (float*)distJ_t.data_ptr();

  // radial and angular share the same data of atomI, startIdxJ and nI
  result.atomI_t = torch::empty(total_atoms * 2, d_options.dtype(torch::kInt32));
  AtomI* atomI_p = (AtomI*)result.atomI_t.data_ptr();
  result.startIdxJ_t = torch::empty(total_atoms, d_options.dtype(torch::kInt32));
  int* startIdxJ_p = (int*)result.startIdxJ_t.data_ptr();

  // radial_num_per_atom ranges from 10 - 60
  result.radialNbr.numJPerI_t = torch::zeros(total_atoms, d_options.dtype(torch::kInt32));
  result.radialNbr.numJPerI_p = (int*)result.radialNbr.numJPerI_t.data_ptr();

  constexpr int ATOM_I_PER_BLOCK = 32;
  if (n_molecules == 1) {
    if (DEBUG_TEST)
      printf("single molecule, %d atoms\n", max_natoms_per_mol);

    constexpr int ATOM_J_PER_SUBTILE = 32;
    constexpr int ATOM_J_PER_TILE = ATOM_I_PER_BLOCK * ATOM_J_PER_SUBTILE;
    int blocks = (total_atoms + ATOM_I_PER_BLOCK - 1) / ATOM_I_PER_BLOCK;
    dim3 block(ATOM_J_PER_SUBTILE, ATOM_I_PER_BLOCK, 1);
    pairwiseDistanceSingleMolecule<ATOM_I_PER_BLOCK, ATOM_J_PER_TILE><<<blocks, block>>>(
        species_t.packed_accessor32<int, 2, torch::RestrictPtrTraits>(),
        coordinates_t.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
        result.radialNbr.numJPerI_t.packed_accessor32<int, 1, torch::RestrictPtrTraits>(),
        atomI_p,
        atomJ_p,
        distJ_p,
        Rcr,
        max_natoms_per_mol);
    result.nI = total_atoms;
  } else {
    // tmp storage
    Tensor numJPerI_t = torch::empty(total_atoms, d_options.dtype(torch::kInt32));
    int* numJPerI_p = (int*)numJPerI_t.data_ptr();
    Tensor atom_i_t = torch::empty(total_atoms * 2, d_options.dtype(torch::kInt32));
    AtomI* atom_i_p = (AtomI*)atom_i_t.data_ptr();

    dim3 block(32, 4, 1);
    // Compute pairwise distance (Rij) for all atom pairs in a molecule
    // maximum 4096 atoms, which needs 49152 byte (48 kb) of shared memory
    // TODO: the kernel is not optimized for batched huge molecule (max_natoms_per_mol > 1000)
    int smem_pairdist = sizeof(float) * max_natoms_per_mol * 5; // x, y, z, spe, counter
    pairwiseDistance<<<n_molecules, block, smem_pairdist, stream>>>(
        species_t.packed_accessor32<int, 2, torch::RestrictPtrTraits>(),
        coordinates_t.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
        numJPerI_t.packed_accessor32<int, 1, torch::RestrictPtrTraits>(),
        atom_i_p,
        atomJ_p,
        distJ_p,
        Rcr,
        max_natoms_per_mol);

    // remove padding atomsI
    result.nI = cubDeviceSelectIf(
        numJPerI_p,
        result.radialNbr.numJPerI_p,
        total_atoms,
        [=] __device__(const int numj) { return (bool)numj; },
        stream);

    // cub::DeviceSelect::Flagged Bug: flag current only allow bool or int which is ether 0 or 1
    // https://github.com/NVIDIA/cub/issues/235
    auto flags_t = numJPerI_t.to(torch::kBool);
    char* flags_p = (char*)flags_t.data_ptr();

    int num_i = cubDeviceSelectFlagged(atom_i_p, atomI_p, total_atoms, flags_p, stream);
  }

  if (DEBUG_TEST)
    printf("num_i %d\n", result.nI);

  cubScan(result.radialNbr.numJPerI_p, startIdxJ_p, total_atoms, stream);
  result.radialNbr.nJ =
      result.startIdxJ_t[total_atoms - 1].item<int>() + result.radialNbr.numJPerI_t[total_atoms - 1].item<int>();
  if (DEBUG_TEST)
    printf("result.radialNbr.nJ %d\n", result.radialNbr.nJ);

  result.radialNbr.atomJ_t = torch::empty(result.radialNbr.nJ, d_options.dtype(torch::kInt32));
  result.radialNbr.atomJ_p = (int*)result.radialNbr.atomJ_t.data_ptr();
  result.radialNbr.distJ_t = torch::empty(result.radialNbr.nJ, d_options.dtype(torch::kFloat32));
  result.radialNbr.distJ_p = (float*)result.radialNbr.distJ_t.data_ptr();

  result.angularNbr.atomJ_t = torch::empty(result.radialNbr.nJ, d_options.dtype(torch::kInt32));
  result.angularNbr.atomJ_p = (int*)result.angularNbr.atomJ_t.data_ptr();
  result.angularNbr.distJ_t = torch::empty(result.radialNbr.nJ, d_options.dtype(torch::kFloat32));
  result.angularNbr.distJ_p = (float*)result.angularNbr.distJ_t.data_ptr();

  result.angularNbr.numJPerI_t = torch::zeros(total_atoms, d_options.dtype(torch::kInt32));
  result.angularNbr.numJPerI_p = (int*)result.angularNbr.numJPerI_t.data_ptr();

  { // cutoffSelect
    int ATOM_J_PER_TILE = 16;
    dim3 block(ATOM_J_PER_TILE, ATOM_I_PER_BLOCK, 1);
    int blocks = (result.nI + ATOM_I_PER_BLOCK - 1) / ATOM_I_PER_BLOCK;
    cutoffSelect<ATOM_I_PER_BLOCK><<<blocks, block>>>(
        atomJ_p,
        distJ_p,
        atomI_p,
        result.radialNbr.atomJ_p,
        result.radialNbr.distJ_p,
        result.angularNbr.atomJ_p,
        result.angularNbr.distJ_p,
        result.radialNbr.numJPerI_p,
        startIdxJ_p,
        Rca,
        result.angularNbr.numJPerI_p,
        result.nI,
        max_natoms_per_mol);

    // TODO angular and radial only need one copy of atom_i, start_j and num_i
    result.angularNbr.nJ = cubSum(result.angularNbr.numJPerI_p, result.nI, stream);
    // result.angularNbr.nJ = at::sum(result.angularNbr.numJPerI_t).item<int>();
    if (DEBUG_TEST)
      printf("result.angularNbr.nJ %d\n", result.angularNbr.nJ);
  }

  { // RadialAEV
    result.radialNbr.maxNumJPerI_aligned = align<4>(cubMax(result.radialNbr.numJPerI_p, result.nI, stream));
    constexpr dim3 block_radial(8, 16, 1);
    int smem_radial = aev_params.radial_length * sizeof(float) + result.radialNbr.maxNumJPerI_aligned * sizeof(float);
    cuRadialAEVs<int, float><<<result.nI, block_radial, smem_radial, stream>>>(
        species_t.packed_accessor32<int, 2, torch::RestrictPtrTraits>(),
        aev_params.ShfR_t.packed_accessor32<float, 1, torch::RestrictPtrTraits>(),
        aev_params.EtaR_t.packed_accessor32<float, 1, torch::RestrictPtrTraits>(),
        result.aev_t.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
        result.radialNbr.atomJ_p,
        result.radialNbr.distJ_p,
        atomI_p,
        result.radialNbr.numJPerI_p,
        startIdxJ_p,
        aev_params.Rcr,
        aev_params.radial_length,
        aev_params.radial_sublength,
        result.radialNbr.nJ,
        result.radialNbr.maxNumJPerI_aligned);
  }

  { // Angular
    auto smem_size = [&aev_params](int max_nbrs, int ncatom_per_tpb) {
      int sm_aev = sizeof(float) * align<4>(aev_params.angular_length); // (angular_length / 4 + 1) * 4
      int sxyz = sizeof(float) * max_nbrs * 3;
      int sRij = sizeof(float) * max_nbrs;
      int sfc = sizeof(float) * max_nbrs;
      int sj = sizeof(int) * max_nbrs;

      // e.g. when max_nbrs is 2
      // ANI1x: (20 * 6 * 4 + 324 * 4) * 2 = 3552
      // ANI2x: (20 * 6 * 4 + 896 * 4) * 2 = 8128
      return (sm_aev + sxyz + sRij + sfc + sj) * ncatom_per_tpb;
    };

    result.angularNbr.maxNumJPerI_aligned = align<4>(cubMax(result.angularNbr.numJPerI_p, result.nI, stream));
    int smem_size_aligned = smem_size(result.angularNbr.maxNumJPerI_aligned, 1);
    int angular_length_aligned = align<4>(aev_params.angular_length);
    if (DEBUG_TEST)
      printf(
          "result.angularNbr.maxNumJPerI_aligned %d -- angular smem_size %d\n",
          result.angularNbr.maxNumJPerI_aligned,
          smem_size_aligned);
    constexpr dim3 block(C10_WARP_SIZE, 4, 1);
    cuAngularAEVs<block.x, block.y><<<result.nI, block, smem_size_aligned, stream>>>(
        species_t.packed_accessor32<int, 2, torch::RestrictPtrTraits>(),
        coordinates_t.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
        aev_params.ShfA_t.packed_accessor32<float, 1, torch::RestrictPtrTraits>(),
        aev_params.ShfZ_t.packed_accessor32<float, 1, torch::RestrictPtrTraits>(),
        aev_params.EtaA_t.packed_accessor32<float, 1, torch::RestrictPtrTraits>(),
        aev_params.Zeta_t.packed_accessor32<float, 1, torch::RestrictPtrTraits>(),
        result.aev_t.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
        result.angularNbr.atomJ_p,
        result.angularNbr.distJ_p,
        atomI_p,
        result.angularNbr.numJPerI_p,
        startIdxJ_p,
        aev_params.Rca,
        aev_params.angular_length,
        aev_params.angular_sublength,
        aev_params.radial_length,
        aev_params.num_species,
        result.angularNbr.maxNumJPerI_aligned,
        angular_length_aligned,
        result.nI);
  }
}

Tensor cuaev_backward(const Tensor& grad_output, const AEVScalarParams& aev_params, const Result& result) {
  using namespace torch::indexing;
  Tensor coordinates_t = result.coordinates_t;
  Tensor species_t = result.species_t;

  const int n_molecules = coordinates_t.size(0);
  const int max_natoms_per_mol = coordinates_t.size(1);
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  auto grad_coord = torch::zeros(coordinates_t.sizes(), coordinates_t.options().requires_grad(false)); // [2, 5, 3]

  AtomI* atomI_p = (AtomI*)result.atomI_t.data_ptr();
  int* d_numPairsPerCenterAtom = (int*)result.angularNbr.numJPerI_t.data_ptr();
  int* d_centerAtomStartIdx = (int*)result.startIdxJ_t.data_ptr();
  int* radial_numJPerI_p = (int*)result.radialNbr.numJPerI_t.data_ptr();
  int* startIdxJ_p = (int*)result.startIdxJ_t.data_ptr();

  Tensor grad_radial_dist = torch::zeros(result.radialNbr.nJ, coordinates_t.options().requires_grad(false));

  int block_size = 64;
  // int nblocks = (result.radialNbr.nJ * 8 + block_size - 1) / block_size;
  constexpr dim3 block_radial(8, 16, 1);
  int smem_radial = result.radialNbr.maxNumJPerI_aligned * sizeof(float) +
      result.radialNbr.maxNumJPerI_aligned * sizeof(float) * 3; // grad_dist, grad_coord
  cuRadialAEVs_backward_or_doublebackward<false, int, float, 8><<<result.nI, block_radial, smem_radial, stream>>>(
      coordinates_t.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
      species_t.packed_accessor32<int, 2, torch::RestrictPtrTraits>(),
      aev_params.ShfR_t.packed_accessor32<float, 1, torch::RestrictPtrTraits>(),
      aev_params.EtaR_t.packed_accessor32<float, 1, torch::RestrictPtrTraits>(),
      grad_output.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
      grad_radial_dist.packed_accessor32<float, 1, torch::RestrictPtrTraits>(), // TODO remove this
      grad_coord.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
      result.radialNbr.atomJ_t.packed_accessor32<int, 1, torch::RestrictPtrTraits>(),
      result.radialNbr.distJ_t.packed_accessor32<float, 1, torch::RestrictPtrTraits>(),
      atomI_p,
      radial_numJPerI_p,
      startIdxJ_p,
      aev_params.Rcr,
      aev_params.radial_length,
      aev_params.radial_sublength,
      result.radialNbr.nJ,
      result.radialNbr.maxNumJPerI_aligned);

  auto smem_size = [&aev_params](int max_nbrs, int ncatom_per_tpb) {
    int sxyz = sizeof(float) * max_nbrs * 3;
    int sj_xyz_grad = sizeof(float) * max_nbrs * 3;
    int sRij = sizeof(float) * max_nbrs;
    int sfc = sizeof(float) * max_nbrs;
    int sfc_grad = sizeof(float) * max_nbrs;
    int sj = sizeof(int) * max_nbrs;

    return (sxyz + sj_xyz_grad + sRij + sfc + sfc_grad + sj) * ncatom_per_tpb;
  };

  block_size = 32;
  const int nthreads_per_catom = 32;
  const int nblocks_angAEV = (result.nI * nthreads_per_catom + block_size - 1) / block_size;
  int smem_size_aligned = smem_size(result.angularNbr.maxNumJPerI_aligned, block_size / nthreads_per_catom);

  // Tensor grad_angular_coord = torch::zeros({result.angularNbr.nJ, 3}, coordinates_t.options().requires_grad(false));
  cuAngularAEVs_backward_or_doublebackward<false><<<nblocks_angAEV, block_size, smem_size_aligned, stream>>>(
      species_t.packed_accessor32<int, 2, torch::RestrictPtrTraits>(),
      coordinates_t.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
      aev_params.ShfA_t.packed_accessor32<float, 1, torch::RestrictPtrTraits>(),
      aev_params.ShfZ_t.packed_accessor32<float, 1, torch::RestrictPtrTraits>(),
      aev_params.EtaA_t.packed_accessor32<float, 1, torch::RestrictPtrTraits>(),
      aev_params.Zeta_t.packed_accessor32<float, 1, torch::RestrictPtrTraits>(),
      grad_output.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
      grad_coord.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
      result.angularNbr.atomJ_t.packed_accessor32<int, 1, torch::RestrictPtrTraits>(),
      result.angularNbr.distJ_t.packed_accessor32<float, 1, torch::RestrictPtrTraits>(),
      atomI_p,
      d_numPairsPerCenterAtom,
      d_centerAtomStartIdx,
      aev_params.Rca,
      aev_params.angular_length,
      aev_params.angular_sublength,
      aev_params.radial_length,
      aev_params.num_species,
      result.angularNbr.maxNumJPerI_aligned,
      align<4>(aev_params.angular_length),
      result.nI);

  return grad_coord;
}

Tensor cuaev_double_backward(const Tensor& grad_force, const AEVScalarParams& aev_params, const Result& result) {
  using namespace torch::indexing;
  Tensor coordinates_t = result.coordinates_t;
  Tensor species_t = result.species_t;

  const int n_molecules = coordinates_t.size(0);
  const int max_natoms_per_mol = coordinates_t.size(1);
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  int aev_length = aev_params.radial_length + aev_params.angular_length;

  auto grad_grad_aev = torch::zeros(
      {coordinates_t.size(0), coordinates_t.size(1), aev_length},
      coordinates_t.options().requires_grad(false)); // [2, 5, 384]

  AtomI* atomI_p = (AtomI*)result.atomI_t.data_ptr();
  int* d_numPairsPerCenterAtom = (int*)result.angularNbr.numJPerI_t.data_ptr();
  int* d_centerAtomStartIdx = (int*)result.startIdxJ_t.data_ptr();
  int* radial_numJPerI_p = (int*)result.radialNbr.numJPerI_t.data_ptr();
  int* startIdxJ_p = (int*)result.startIdxJ_t.data_ptr();

  auto grad_force_coord_Rij = torch::zeros({result.radialNbr.nJ}, coordinates_t.options().requires_grad(false));

  int block_size = 64;
  constexpr dim3 block_radial(8, 16, 1);
  int smem_radial = result.radialNbr.maxNumJPerI_aligned * sizeof(float) +
      aev_params.radial_length * sizeof(float); // grad_dist, grad_grad_aev
  cuRadialAEVs_backward_or_doublebackward<true, int, float, 8><<<result.nI, block_radial, smem_radial, stream>>>(
      coordinates_t.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
      species_t.packed_accessor32<int, 2, torch::RestrictPtrTraits>(),
      aev_params.ShfR_t.packed_accessor32<float, 1, torch::RestrictPtrTraits>(),
      aev_params.EtaR_t.packed_accessor32<float, 1, torch::RestrictPtrTraits>(),
      grad_grad_aev.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
      grad_force_coord_Rij.packed_accessor32<float, 1, torch::RestrictPtrTraits>(),
      grad_force.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
      result.radialNbr.atomJ_t.packed_accessor32<int, 1, torch::RestrictPtrTraits>(),
      result.radialNbr.distJ_t.packed_accessor32<float, 1, torch::RestrictPtrTraits>(),
      atomI_p,
      radial_numJPerI_p,
      startIdxJ_p,
      aev_params.Rcr,
      aev_params.radial_length,
      aev_params.radial_sublength,
      result.radialNbr.nJ,
      result.radialNbr.maxNumJPerI_aligned);

  auto smem_size = [&aev_params](int max_nbrs, int ncatom_per_tpb) {
    int sxyz = sizeof(float) * max_nbrs * 3;
    int sj_xyz_grad = sizeof(float) * max_nbrs * 3;
    int sRij = sizeof(float) * max_nbrs;
    int sfc = sizeof(float) * max_nbrs;
    int sfc_grad = sizeof(float) * max_nbrs;
    int sj = sizeof(int) * max_nbrs;

    return (sxyz + sj_xyz_grad + sRij + sfc + sfc_grad + sj) * ncatom_per_tpb;
  };

  block_size = 32;
  const int nthreads_per_catom = 32;
  const int nblocks_angAEV = (result.nI * nthreads_per_catom + block_size - 1) / block_size;
  int smem_size_aligned = smem_size(result.angularNbr.maxNumJPerI_aligned, block_size / nthreads_per_catom);

  cuAngularAEVs_backward_or_doublebackward<true><<<nblocks_angAEV, block_size, smem_size_aligned, stream>>>(
      species_t.packed_accessor32<int, 2, torch::RestrictPtrTraits>(),
      coordinates_t.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
      aev_params.ShfA_t.packed_accessor32<float, 1, torch::RestrictPtrTraits>(),
      aev_params.ShfZ_t.packed_accessor32<float, 1, torch::RestrictPtrTraits>(),
      aev_params.EtaA_t.packed_accessor32<float, 1, torch::RestrictPtrTraits>(),
      aev_params.Zeta_t.packed_accessor32<float, 1, torch::RestrictPtrTraits>(),
      grad_force.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
      grad_grad_aev.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
      result.angularNbr.atomJ_t.packed_accessor32<int, 1, torch::RestrictPtrTraits>(),
      result.angularNbr.distJ_t.packed_accessor32<float, 1, torch::RestrictPtrTraits>(),
      atomI_p,
      d_numPairsPerCenterAtom,
      d_centerAtomStartIdx,
      aev_params.Rca,
      aev_params.angular_length,
      aev_params.angular_sublength,
      aev_params.radial_length,
      aev_params.num_species,
      result.angularNbr.maxNumJPerI_aligned,
      align<4>(aev_params.angular_length),
      result.nI);

  return grad_grad_aev;
}