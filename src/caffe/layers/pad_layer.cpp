#include <algorithm>
#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layer_factory.hpp"
#include "caffe/layers/pad_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void PadLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
  const vector<Blob<Dtype>*>& top) {
  CHECK_NE(top[0], bottom[0]) << this->type() << " Layer does not "
    "allow in-place computation.";

  pad_u_ = this->layer_param_.pad_param().pad_u();
  // CHECK_GE(pad_u_, 0) << "pad_u must >= 0!";

  pad_d_ = this->layer_param_.pad_param().pad_d();
  // CHECK_GE(pad_d_, 0) << "pad_d must >= 0!";

  pad_l_ = this->layer_param_.pad_param().pad_l();
  // CHECK_GE(pad_l_, 0) << "pad_l must >= 0!";

  pad_r_ = this->layer_param_.pad_param().pad_r();
  // CHECK_GE(pad_r_, 0) << "pad_r must >= 0!";

  pad_value_ = this->layer_param_.pad_param().pad_value();
}

template <typename Dtype>
void PadLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
  const vector<Blob<Dtype>*>& top) {

  int num_axes = bottom[0]->num_axes();

  if(num_axes == 4) {  
      vector<int> top_shape = bottom[0]->shape();
      top_shape[2] = top_shape[2] + pad_u_ + pad_d_;
      CHECK_GE(top_shape[2], 0) << "top_shape[2] must >= 0!";          
      top_shape[3] = top_shape[3] + pad_l_ + pad_r_;
      CHECK_GE(top_shape[3], 0) << "top_shape[3] must >= 0!";      
      top[0]->Reshape(top_shape);
  } else if(num_axes == 3){
      vector<int> top_shape = bottom[0]->shape();
      top_shape[1] = top_shape[1] + pad_u_ + pad_d_;
      CHECK_GE(top_shape[1], 0) << "top_shape[1] must >= 0!";          
      top_shape[2] = top_shape[2] + pad_l_ + pad_r_;
      CHECK_GE(top_shape[2], 0) << "top_shape[2] must >= 0!";          
      top[0]->Reshape(top_shape);
  } else if(num_axes == 2){
      vector<int> top_shape = bottom[0]->shape();
      top_shape[0] = top_shape[0] + pad_u_ + pad_d_;
      CHECK_GE(top_shape[0], 0) << "top_shape[0] must >= 0!";       
      top_shape[1] = top_shape[1] + pad_l_ + pad_r_;
      CHECK_GE(top_shape[1], 0) << "top_shape[1] must >= 0!";       
      top[0]->Reshape(top_shape);
  }
}

template <typename Dtype>
void PadLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
  const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();

  int num = bottom[0]->num();
  int channels = bottom[0]->channels();
  int bottom_height = bottom[0]->height();
  int bottom_width = bottom[0]->width();
  int top_height = top[0]->height();
  int top_width = top[0]->width();
  if(pad_u_ < 0 && pad_d_ < 0 && pad_l_ < 0 && pad_r_ < 0)
  {
    for (int n = 0; n< num; ++n){
      for (int c = 0; c< channels; ++c){
        for (int h = 0; h < top_height; ++h){
          caffe_copy(top_width,
                    bottom_data + bottom[0]->offset(n, c, h + abs(pad_u_), abs(pad_l_)),
                    top_data + top[0]->offset(n, c, h, 0));
        }
      }
    }
  }
  else
  {
    for (int n = 0; n < num; ++n) {
      for (int c = 0; c < channels; ++c) {
        // First copy the main body into place
        for (int h = 0; h < bottom_height; ++h) {
          // copy the width part
          caffe_copy(bottom_width,
                    bottom_data + bottom[0]->offset(n, c, h, 0),
                    top_data + top[0]->offset(n, c, h + pad_u_, pad_l_));
        }
        switch (this->layer_param_.pad_param().pad_type()) {
          case PadParameter_PadType_CONSTANT:
              {
              // Left and right. Loop over the rows not in the vertical padding
              for (int h = pad_u_; h < top_height - pad_d_; ++h) {
                // Offset to current row start (in padding of this row)
                int off = top[0]->offset(n, c, h, 0);
                // Left pad
                for (int wdst = 0; wdst < pad_l_; ++wdst) {
                  *(top_data + off + wdst) = static_cast<Dtype>(pad_value_);
                }
                // Right
                for (int wdst = top_width -pad_r_; wdst < top_width; ++wdst) {
                  *(top_data + off + wdst) = static_cast<Dtype>(pad_value_);
                }
              }
              // Top
              for (int h = 0; h < pad_u_; ++h) {
                int off = top[0]->offset(n, c, h, 0);
                std::fill(top_data + off, top_data + off + top_width,
                          static_cast<Dtype>(pad_value_));
              }
              // Bottom
              for (int h = top_height - pad_d_; h < top_height; ++h) {
                int off = top[0]->offset(n, c, h, 0);
                std::fill(top_data + off, top_data + off + top_width,
                          static_cast<Dtype>(pad_value_));
              }
            }
            break;
          case PadParameter_PadType_REPLICATE:
            {
              // Left and right. Loop over the rows not in the vertical padding
              for (int h = pad_u_; h < top_height - pad_d_; ++h) {
                // Offset to current row start (in padding of this row)
                int off = top[0]->offset(n, c, h, 0);
                const Dtype lval = *(top_data + off + pad_l_),
                  rval = *(top_data + off + top_width - 1 - pad_r_);
                // Left
                for (int wdst = 0; wdst < pad_l_; ++wdst) {
                  *(top_data + off + wdst) = lval;
                }
                // Right
                for (int wdst = top_width - pad_r_; wdst < top_width; ++wdst) {
                  *(top_data + off + wdst) = rval;
                }
              }
              // Top
              // Beginning of this image's data, including padding
              Dtype * dstptr = top_data + top[0]->offset(n, c, 0, 0);
              // First row not in the vertical padding
              Dtype * srcptr = dstptr + pad_u_ * top_width;
              for (int h = 0; h < pad_u_; ++h) {
                std::copy(srcptr, srcptr + top_width,
                          dstptr + h * top_width);
              }
              // Bottom
              // Start of last row not in the vertical padding
              srcptr = top_data + top[0]->offset(n, c, top_height - 1 - pad_d_, 0);
              // Start of first row in bottom padding
              dstptr = srcptr + top_width;
              for (int h = 0; h < pad_d_; ++h) {
                std::copy(srcptr, srcptr + top_width,
                          dstptr + h * top_width);
              }
            }
            break;
          case PadParameter::REFLECT:
            {
              /*
              *************
              **xxxxxxxxx**
              **xxxxxxxxx**
              **xxxxxxxxx**
              **xxxxxxxxx**
              *************
              */
              // Left and right. Loop over the rows not in the vertical padding
              for (int h = pad_u_; h < top_height - pad_d_; ++h) {
                // Offset to current row start (in padding of this row)
                int off = top[0]->offset(n, c, h, 0);
                // Left
                for (int wdst = pad_l_ -1, wsrc = pad_l_ + 1; wdst >= 0; --wdst, ++wsrc) {
                  *(top_data + off + wdst) = *(top_data + off + wsrc);
                }
                // Right
                for (int wdst = top_width - pad_r_, wsrc = wdst-2; wdst < top_width;
                    ++wdst, --wsrc) {
                  *(top_data + off + wdst) = *(top_data + off + wsrc);
                }
              }
              /*
              *************
              xxxxxxxxxxxxx
              xxxxxxxxxxxxx
              xxxxxxxxxxxxx
              xxxxxxxxxxxxx
              *************
              */              
              // Top
              for (int hdst = pad_u_-1, hsrc = pad_u_+1; hdst >= 0; --hdst, ++hsrc) {
                caffe_copy(top_width,
                          top_data + top[0]->offset(n, c, hsrc, 0),
                          top_data + top[0]->offset(n, c, hdst, 0));
              }
              /*
              xxxxxxxxxxxxx
              xxxxxxxxxxxxx
              xxxxxxxxxxxxx
              xxxxxxxxxxxxx
              xxxxxxxxxxxxx
              *************
              */                 
              // Bottom
              for (int hdst = top_height - pad_d_, hsrc = hdst-2; hdst < top_height;
                  ++hdst, --hsrc) {
                caffe_copy(top_width,
                          top_data + top[0]->offset(n, c, hsrc, 0),
                          top_data + top[0]->offset(n, c, hdst, 0));
              }
              /*
              xxxxxxxxxxxxx
              xxxxxxxxxxxxx
              xxxxxxxxxxxxx
              xxxxxxxxxxxxx
              xxxxxxxxxxxxx
              xxxxxxxxxxxxx
              */               
            }
            break;        
        }
      }
    }
  }
}
template <typename Dtype>
void PadLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& bottom, 
  const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& top) {
}
INSTANTIATE_CLASS(PadLayer);
REGISTER_LAYER_CLASS(Pad);

}  // namespace caffe
