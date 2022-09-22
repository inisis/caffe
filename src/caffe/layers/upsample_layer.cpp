#include <vector>
#include "caffe/layers/upsample_layer.hpp"

namespace caffe {
    
    template <typename Dtype>
    void UpsampleLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
        UpsampleParameter upsample_param = this->layer_param_.upsample_param();
        scale_ = upsample_param.scale();
        dims_num = upsample_param.dims_num();
    }
    
    template <typename Dtype>
    void UpsampleLayer<Dtype>::Reshape(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
        vector<int> out_shape;
        for (int i = 0; i < bottom[0]->num_axes(); i++) {
            out_shape.push_back(bottom[0]->shape(i));
        }
        if (dims_num == 4)
        {
            out_shape[bottom[0]->num_axes() - 1] *= scale_;
            out_shape[bottom[0]->num_axes() - 2] *= scale_;
        }else if ( dims_num == 3 )
        {
            out_shape[bottom[0]->num_axes() - 1] *= scale_;
        }
        else
        {
            LOG(FATAL) << "dims_num JUST SURPORT 3 or 4";
        }
        top[0]->Reshape(out_shape);
        }
    
    template <typename Dtype>
    void UpsampleLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
        if ( dims_num == 4)//NCHW
        {
            int N = top[0]->shape(0);
            int C = top[0]->shape(1);
            int H = top[0]->shape(2);
            int W = top[0]->shape(3);            
            const Dtype *input = bottom[0]->cpu_data();
            Dtype *output = top[0]->mutable_cpu_data();
            for (int n = 0; n < N; n++) {
                for (int c = 0; c < C; c++) {
                    for (int h = 0; h < H; h++) {
                        for (int w = 0; w < W; w++) {
                            int nw = w/scale_;
                            int nh = h/scale_;
                            int out_idx = (((n * C + c) * H) + h) * W + w;
                            int in_idx = (((n * C + c) * (H / scale_)) + nh) * (W / scale_) + nw;
                            output[out_idx] = input[in_idx];
                        }
                    }
                }
            }
        }
        else if ( dims_num == 3)//NCW
        {
            int N = top[0]->shape(0);
            int C = top[0]->shape(1);
            int W = top[0]->shape(2);
            const Dtype *input = bottom[0]->cpu_data();
            Dtype *output = top[0]->mutable_cpu_data();
            for (int n = 0; n < N; n++) {
                for (int c = 0; c < C; c++) {
                    for (int w = 0; w < W; w++) {
                        int nw = w/scale_;
                        int out_idx = n * C * W + c * W + w;
                        int in_idx = n * C * (W/scale_) + c * (W/scale_) + nw;
                        output[out_idx] = input[in_idx];
                    }
                }
            }
        }
        else
        {
            LOG(FATAL) << "dims_num JUST SURPORT 3 or 4";
        }
    }
    
    #ifdef CPU_ONLY
    STUB_GPU(UpsampleLayer);
    #endif
    
    INSTANTIATE_CLASS(UpsampleLayer);
    REGISTER_LAYER_CLASS(Upsample);
}  // namespace caffe
