#ifndef CAFFE_BILINEAR_INTERPOLATE_LAYER_HPP_
#define CAFFE_BILINEAR_INTERPOLATE_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe{
    
    template <typename Dtype>
    class BilinearInterpolateLayer : public Layer<Dtype>{
        public:
        explicit BilinearInterpolateLayer(const LayerParameter& param)
            :Layer<Dtype>(param) {}
        virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
            const vector<Blob<Dtype>*>& top);
        virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
            const vector<Blob<Dtype>*>& top);
            
        virtual inline const char* type() const {return "BilinearInterpolate";}
        virtual inline int MinBottomBlobs() const {return 1;}
        virtual inline int MaxBottomBlobs() const { return 1;}
        virtual inline int ExactNumTopBlobs() const {return 1;}
        
        virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
            const vector<Blob<Dtype>*>& top);
        //virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
        //    const vector<Blob<Dtype>*>& top);
        virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
            const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
        //virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
        //    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
            
        private:
        int dst_h_, dst_w_;
        float scale_factor_;
        bool align_corners_;
        Blob<Dtype> dst_h_temp_, dst_w_temp_;
        Blob<Dtype> h_temp_, w_temp_;
        Blob<Dtype> expand_h_help_, expand_w_help_;
        Blob<Dtype> h_, w_;
        Blob<Dtype> h0_,h1_,w0_,w1_;
    };

}  // namespace caffe
#endif  // CAFFE_BILINEAR_INTERPOLATE_LAYER_HPP_
