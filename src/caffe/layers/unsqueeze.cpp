#include <vector>

#include "caffe/layers/unsqueeze.hpp"

namespace caffe {

template <typename Dtype>
void UnsqueezeLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
  const UnsqueezeParameter& unsqueeze_param = this->layer_param_.unsqueeze_param();
  vector<int> bottom_shape = bottom[0]->shape();
  int num = bottom[0]->num_axes();
  if(!unsqueeze_param.has_dim()) {
      LOG(FATAL)<<"dim must be set.";
  } else {
      int index = unsqueeze_param.dim();
      if(index < 0){
          index = index + num + 1;
      }
      if(index < 0 || index > num){
          LOG(FATAL)<<"dim:( "<< index <<") overtake the dims of bottom.";
      }
      for(int i=0; i<num; i++){
          int dim = bottom_shape[i];
          if(i == index) top_shape_.push_back(1);
          top_shape_.push_back(dim);
      }
      if(index == num) top_shape_.push_back(1);
  }
}

template <typename Dtype>
void UnsqueezeLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
    top[0]->Reshape(top_shape_);
}

template <typename Dtype>
void UnsqueezeLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
    caffe_copy(bottom[0]->count(), bottom[0]->cpu_data(), top[0]->mutable_cpu_data());
}

template <typename Dtype>
void UnsqueezeLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {

}


#ifdef CPU_ONLY
//STUB_GPU(UnsqueezeLayer);
#endif

INSTANTIATE_CLASS(UnsqueezeLayer);
REGISTER_LAYER_CLASS(Unsqueeze);

} //namespace caffe
