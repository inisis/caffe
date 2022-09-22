#include <algorithm>
#include <vector>

#include "caffe/layers/hardsigmoid_layer.hpp"

namespace caffe {

template <typename Dtype>
void HardSigmoidLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  HardSigmoidParameter hardsigmoid_param = this->layer_param_.hardsigmoid_param();
  
  const float alpha = hardsigmoid_param.alpha();
  const float beta = hardsigmoid_param.beta();

  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  const int count = bottom[0]->count();
  for (int i = 0; i < count; ++i) {
    top_data[i] = std::max(Dtype(0),std::min(Dtype(1), (bottom_data[i] * alpha + beta)));
  }
}

#ifdef CPU_ONLY
STUB_GPU(HardSigmoidLayer);
#endif

INSTANTIATE_CLASS(HardSigmoidLayer);
REGISTER_LAYER_CLASS(HardSigmoid);
}  // namespace caffe

