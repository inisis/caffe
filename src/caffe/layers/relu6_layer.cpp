#include <algorithm>
#include <vector>

#include "caffe/layers/relu6_layer.hpp"

namespace caffe {

template <typename Dtype>
void ReLU6Layer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  const int count = bottom[0]->count();
  Dtype negative_slope = this->layer_param_.relu6_param().negative_slope();
  Dtype threshold = this->layer_param_.relu6_param().threshold();
  for (int i = 0; i < count; ++i) {
    top_data[i] =std::min(std::max(bottom_data[i], Dtype(0))
        + negative_slope * std::min(bottom_data[i], Dtype(0)), threshold);
  }
}

template <typename Dtype>
void ReLU6Layer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[0]) {
    const Dtype* bottom_data = bottom[0]->cpu_data();
    const Dtype* top_diff = top[0]->cpu_diff();
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    const int count = bottom[0]->count();
    Dtype negative_slope = this->layer_param_.relu6_param().negative_slope();
    Dtype threshold = this->layer_param_.relu6_param().threshold();
    for (int i = 0; i < count; ++i) {
      bottom_diff[i] = (top_diff[i] * ((bottom_data[i] > 0)
          + negative_slope * (bottom_data[i] <= 0)))*(bottom_data[i] < threshold);
    }
  }
}


#ifdef CPU_ONLY
STUB_GPU(ReLU6Layer);
#endif

INSTANTIATE_CLASS(ReLU6Layer);
REGISTER_LAYER_CLASS(ReLU6);
}  // namespace caffe

