#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layers/maxunpool_layer.hpp"

namespace caffe {

template <typename Dtype>
void MaxUnPoolLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  MaxUnPoolParameter max_unpool_param = this->layer_param_.max_unpool_param();

  CHECK((max_unpool_param.has_dst_h() && max_unpool_param.has_dst_w()))
      << "dst_h and dst_w are required.";
  unpooled_height_ = max_unpool_param.dst_h();
  unpooled_width_ = max_unpool_param.dst_w();
}

template <typename Dtype>
void MaxUnPoolLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  CHECK_EQ(4, bottom[0]->num_axes()) << "Input must have 4 axes, "
      << "corresponding to (num, channels, height, width)";
  channels_ = bottom[0]->channels();
  height_ = bottom[0]->height();
  width_ = bottom[0]->width();

  top[0]->Reshape(bottom[0]->num(), channels_, unpooled_height_,
      unpooled_width_);
}

template <typename Dtype>
void MaxUnPoolLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  const Dtype* bottom_mask_data = bottom[1]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  const int top_count = top[0]->count();
  caffe_set(top_count, Dtype(0), top_data);
  // The main loop
  for (int n = 0; n < bottom[0]->num(); ++n) {
    for (int c = 0; c < channels_; ++c) {
      for (int i = 0; i < height_ * width_; ++i) {
        const int idx = static_cast<int>(bottom_mask_data[i]);
        if (idx >= unpooled_height_ * unpooled_width_) {
          LOG(FATAL) << "uppool top index " << idx << " out of range - ";
        }
        top_data[idx] = bottom_data[i];
      }
      // compute offset
      bottom_data += bottom[0]->offset(0, 1);
      bottom_mask_data += bottom[1]->offset(0, 1);
      top_data += top[0]->offset(0, 1);
    }
  }
}

#ifdef CPU_ONLY
STUB_GPU(MaxUnPoolLayer);
#endif

INSTANTIATE_CLASS(MaxUnPoolLayer);
REGISTER_LAYER_CLASS(MaxUnPool);
}  // namespace caffe
