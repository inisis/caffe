#ifndef CAFFE_UNSQUEEZE_LAYER_HPP_
#define CAFFE_UNSQUEEZE_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

template <typename Dtype>
class UnsqueezeLayer:public Layer<Dtype> {
  public:
    explicit UnsqueezeLayer(const LayerParameter& param)
	    :Layer<Dtype>(param) {}
    virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
		    const vector<Blob<Dtype>*>& top);
    virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
		    const vector<Blob<Dtype>*>& top);

    virtual inline const char* type() const {return "Unsqueeze";}
    virtual inline int ExactNumBottomBlobs() const {return 1;}
    virtual inline int ExactNumTopBlobs() const {return 1;}

  protected:
    virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
		    const vector<Blob<Dtype>*>& top);
    virtual void Backward_cpu(const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down,
		    const vector<Blob<Dtype>*>& bottom);

    vector<int> top_shape_;

};

} // namespace caffe

#endif //CAFFE_UNSQUEEZE_LAYER_HPP_
