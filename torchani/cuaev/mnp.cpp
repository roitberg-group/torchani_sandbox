#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <torch/extension.h>

using torch::Tensor;
using torch::autograd::AutogradContext;
using torch::autograd::tensor_list;

#define USE_STREAMS

class MultiNetFunction : public torch::autograd::Function<MultiNetFunction> {
 public:
  static Tensor forward(
      AutogradContext* ctx,
      Tensor aev,
      int64_t num_networks,
      tensor_list idx_list,
      std::vector<std::vector<Tensor>> weight_list,
      std::vector<std::vector<Tensor>> bias_list,
      std::vector<at::Stream> stream_list) {
    tensor_list to_save;
    std::vector<int64_t> num_layers_list(num_networks, 0);
    std::vector<at::Tensor> outputs;
    Tensor energy_list = at::zeros(num_networks, aev.options());

#ifdef USE_STREAMS
    at::cuda::CUDAStream current_stream = c10::cuda::getCurrentCUDAStream();
    cudaEvent_t start_event;
    cudaEventCreate(&start_event);
    cudaEventRecord(start_event, current_stream);
    std::vector<cudaEvent_t> event_list;
    for (int i = 0; i < idx_list.size(); i++) {
      cudaEvent_t tmp_evt;
      cudaEventCreate(&tmp_evt);
      event_list.push_back(tmp_evt);
    }
#endif

    // loop over networks
    for (int i = 0; i < idx_list.size(); i++) {
      // only run if species idx is not empty
      if (idx_list[i].size(0) > 0) {
#ifdef USE_STREAMS
        cudaStreamWaitEvent(c10::cuda::CUDAStream(stream_list[i]), start_event, 0);
        at::cuda::CUDAStreamGuard guard(stream_list[i]);
#endif
        int num_layers = weight_list[i].size();
        Tensor input_ = aev.index_select(0, idx_list[i]);

        // loop over layers
        for (int j = 0; j < num_layers; j++) {
          // linear layer
          input_ = at::addmm(bias_list[i][j], input_, weight_list[i][j]);
          // activation layer if it's not the last layer
          if (j < num_layers - 1) {
            input_ = at::celu_(input_, 0.1);
          }
          // number of layers counter of current network for backward
          num_layers_list[i]++;
          // save weight and intermediate for backward
          to_save.push_back(weight_list[i][j]);
          to_save.push_back(input_);
        }

        // sum out without cudaMemcpyAsync
        auto tmp_energy = energy_list[i];
        at::sum_out(tmp_energy, input_.view(-1), 0, /* keepdim */ false);
#ifdef USE_STREAMS
        cudaEventRecord(event_list[i], c10::cuda::CUDAStream(stream_list[i]));
#endif
      }
    }

    // save species index for backward
    for (int i = 0; i < num_networks; i++) {
      to_save.push_back(idx_list[i]);
    }

#ifdef USE_STREAMS
    // default stream waits until all stream finish
    for (int i = 0; i < num_networks; i++) {
      if (idx_list[i].size(0) > 0)
        cudaStreamWaitEvent(current_stream, event_list[i], 0);
    }
#endif

    to_save.push_back(aev);
    ctx->save_for_backward(to_save);
    ctx->saved_data["num_layers_list"] = c10::List<int64_t>(num_layers_list);
    ctx->saved_data["stream_list"] = c10::List<at::Stream>(stream_list);

    return at::sum(energy_list, 0, true).view({1, 1});
  }

  static tensor_list backward(AutogradContext* ctx, tensor_list grad_o) {
    tensor_list saved_tensors = ctx->get_saved_variables();
    float alpha = 0.1f;
    int num_saved = saved_tensors.size();
    Tensor aev = saved_tensors[num_saved - 1];
    Tensor aev_grad = torch::zeros_like(aev);
    c10::List<int64_t> num_layers_list = ctx->saved_data["num_layers_list"].toIntList();
    c10::List<at::Stream> stream_list = ctx->saved_data["stream_list"].to<c10::List<at::Stream>>();
    int idx = 0;
    tensor_list idx_list;
    int num_networks = num_layers_list.size();
    int idx_list_start_idx = num_saved - 1 - num_networks;

#ifdef USE_STREAMS
    at::cuda::CUDAStream current_stream = c10::cuda::getCurrentCUDAStream();
    cudaEvent_t start_event;
    cudaEventCreate(&start_event);
    cudaEventRecord(start_event, current_stream);
    std::vector<cudaEvent_t> event_list;
    for (int i = 0; i < num_networks; i++) {
      cudaEvent_t tmp_evt;
      cudaEventCreate(&tmp_evt);
      event_list.push_back(tmp_evt);
    }
#endif

    for (int i = 0; i < num_networks; i++) {
      idx_list.push_back(saved_tensors[idx_list_start_idx + i]);
    }

    // loop over networks
    for (int i = 0; i < num_layers_list.size(); i++) {
      // only run if species idx is not empty
      if (idx_list[i].size(0) > 0) {
#ifdef USE_STREAMS
        cudaStreamWaitEvent(c10::cuda::CUDAStream(stream_list[i]), start_event, 0);
        at::cuda::CUDAStreamGuard guard(stream_list[i]);
#endif
        Tensor input_ = grad_o[0].expand({idx_list[i].size(0), -1});

        // loop over layers reversely
        for (int j = num_layers_list[i] - 1; j >= 0; j--) {
          Tensor weight = saved_tensors[(idx + j) * 2].transpose(0, 1);
          Tensor intermediate_result = saved_tensors[(idx + j) * 2 + 1];
          // activation layer backward if it's not the last layer
          if (j < num_layers_list[i] - 1) {
            input_ = at::elu_backward(input_, alpha, 1, 1.0 / alpha, /* is_result */ true, intermediate_result);
          }
          // linear layer backward
          input_ = at::matmul(input_, weight);
        }
        aev_grad.index_put_({idx_list[i]}, input_);
        idx += num_layers_list[i];
#ifdef USE_STREAMS
        cudaEventRecord(event_list[i], c10::cuda::CUDAStream(stream_list[i]));
#endif
      }
    }

#ifdef USE_STREAMS
    // default stream waits until all stream finish
    for (int i = 0; i < num_networks; i++) {
      if (idx_list[i].size(0) > 0)
        cudaStreamWaitEvent(current_stream, event_list[i], 0);
    }
#endif

    return {aev_grad, Tensor(), Tensor(), Tensor(), Tensor(), Tensor()};
  }
};

Tensor run_autograd(
    Tensor aev,
    int64_t num_networks,
    tensor_list idx_list,
    std::vector<std::vector<Tensor>> weight_list,
    std::vector<std::vector<Tensor>> bias_list,
    std::vector<at::Stream> stream_list) {
  return MultiNetFunction::apply(aev, num_networks, idx_list, weight_list, bias_list, stream_list);
}

// mnp stands for multi network parallel
TORCH_LIBRARY(mnp, m) {
  m.def("run", run_autograd);
}

TORCH_LIBRARY_IMPL(mnp, Autograd, m) {
  m.impl("run", run_autograd);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {}
