#ifndef RESIZE_LAYERS_HPP_
#define RESIZE_LAYERS_HPP_

#include <string>
#include <utility>
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
//#include "caffe/common_layers.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {
/**
  * @brief ResizeLayer resize blob using bilinear interpolation
  *
  * 	   In layer configuration, <code>is_pyramid_test</code> specifiy whether
  * 	   the output height and width is fixed. if <code> is_pyramid_test == false </code>,
  * 	   you should give the <code>height</code> and <code>width</code> for output.
  * 	   Otherwise, you should give both <code>out_height_scale</code> and
  * 	   <code>out_width_scale</code>
  */
template <typename Dtype>
class ResizeLayer : public Layer<Dtype>{
public:
	explicit ResizeLayer(const LayerParameter& param)
	: Layer<Dtype>(param){}
	virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
	  const vector<Blob<Dtype>*>& top);
	virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
	  const vector<Blob<Dtype>*>& top);

	virtual inline const char* type() const { return "Resize"; }

	virtual inline int ExactNumTopBlobs() const { return 1; }
	virtual inline int ExactNumBottomBlobs() const { return 1; }

protected:
	virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
	      const vector<Blob<Dtype>*>& top);
	virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
	      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
	virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
		      const vector<Blob<Dtype>*>& top);
	virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
		      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
	vector<Blob<Dtype>*> locs_;
	int out_height_;
	int out_width_;
	int out_channels_;
	int out_num_;

};


}  // namespace caffe

#endif  // RESIZE_LAYERS_HPP_
