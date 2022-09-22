#include <vector>
#include "caffe/layers/pixelshuffle_layer.hpp"

namespace caffe {
    
    template <typename Dtype>
    void PixelShuffleLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
        PixelShuffleParameter pixelshuffle_param = this->layer_param_.pixelshuffle_param();
        upscale_factor_ = pixelshuffle_param.upscale_factor();
    }
    
    template <typename Dtype>
    void PixelShuffleLayer<Dtype>::Reshape(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
        vector<int> out_shape;
        for (int i = 0; i < bottom[0]->num_axes(); i++) {
            out_shape.push_back(bottom[0]->shape(i));
        }
        CHECK_EQ(bottom[0]->shape(1) % upscale_factor_, 0);
        out_shape[bottom[0]->num_axes() - 3] = out_shape[bottom[0]->num_axes() - 3] / (upscale_factor_ * upscale_factor_);
        out_shape[bottom[0]->num_axes() - 2] *= upscale_factor_;
        out_shape[bottom[0]->num_axes() - 1] *= upscale_factor_;
       
        top[0]->Reshape(out_shape);
        }
    
    template <typename Dtype>
    void PixelShuffleLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
        int batch = top[0]->shape(0);
        int channel_out = top[0]->shape(1);
        int height_in = bottom[0]->shape(2);
        int width_in = bottom[0]->shape(3);              
        const Dtype *bottom_data = bottom[0]->cpu_data();
        Dtype *top_data = top[0]->mutable_cpu_data();
        
        for (int n = 0; n < batch; n++)
        {
            for (int p = 0; p < channel_out; p++)
            {
                for (int sh = 0; sh < upscale_factor_; sh++)
                {
                    for (int sw = 0; sw < upscale_factor_; sw++)
                    {
                        int q = p * upscale_factor_ * upscale_factor_ + sh * upscale_factor_ + sw;

                        const Dtype* sptr = bottom_data + bottom[0]->offset(n, q, 0, 0);

                        for (int i = 0; i < height_in; i++)
                        {
                            Dtype* outptr = top_data + top[0]->offset(n, p, i * upscale_factor_ + sh, sw);
                            for (int j = 0; j < width_in; j++)
                            {
                                outptr[0] = sptr[0];

                                sptr++;
                                outptr += upscale_factor_;
                            }
                        }
                    }
                }
            }
        }
    }
    
    #ifdef CPU_ONLY
    STUB_GPU(PixelShuffleLayer);
    #endif
    
    INSTANTIATE_CLASS(PixelShuffleLayer);
    REGISTER_LAYER_CLASS(PixelShuffle);
}  // namespace caffe
