#ifndef CAFFE_PAD_LAYER_HPP_
#define CAFFE_PAD_LAYER_HPP_

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

/*
* @brief zero-padding to up down left right of bottom
*
* Note: Back-propagate just drop the pad derivatives
*/
template <typename Dtype>
class PadLayer : public Layer<Dtype> {
  public:
    explicit PadLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
    virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
    virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

    virtual inline const char* type() const { return "Pad"; }
    virtual inline int ExactNumBottomBlobs() const { return 1; }
    virtual inline int ExactNumTopBlobs() const { return 1; }

  protected:
    virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
    virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
    PadParameter_PadType PAD_TYPE_;      
    int pad_u_, pad_d_, pad_l_, pad_r_;
    Dtype pad_value_;
};

}  // namespace caffe

#endif  // CAFFE_PAD_LAYER_HPP_
