#include <algorithm>
#include <vector>

#include "caffe/layers/hardswish_layer.hpp"

namespace caffe {

template <typename Dtype>
void HardSwishLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  const int count = bottom[0]->count();
  for (int i = 0; i < count; ++i) {
    top_data[i] = bottom_data[i]  *(std::max(Dtype(0),std::min(Dtype(1), bottom_data[i]/Dtype(6.0) + Dtype(0.5))));
  }
}

#ifdef CPU_ONLY
STUB_GPU(HardSwishLayer);
#endif

INSTANTIATE_CLASS(HardSwishLayer);
REGISTER_LAYER_CLASS(HardSwish);
}  // namespace caffe

