#include <vector>
#include "caffe/layers/bilinear_interpolate_layer.hpp"

namespace caffe {
    
    template <typename Dtype>
    void BilinearInterpolateLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
        BilinearInterpolateParameter param = this->layer_param_.bilinear_interpolate_param();
        // only dst_size
	if((param.has_dst_h() || param.has_dst_w()) && !param.has_scale_factor()){
	    // dst_h
	    if(param.has_dst_h()){
	        dst_h_ = param.dst_h();
	    }else{
	        dst_h_ = bottom[0]->height();
	    }
            // dst_w
	    if(param.has_dst_w()){
	        dst_w_ = param.dst_w();
	    }else{
	        dst_w_ = bottom[0]->width();
	    }
	}
        // only scale_factor
	if((!param.has_dst_h() && !param.has_dst_w()) && param.has_scale_factor()){
	    scale_factor_ = param.scale_factor();
	    dst_h_ = int(bottom[0]->height() * scale_factor_);
	    dst_w_ = int(bottom[0]->width() * scale_factor_);
	}

        // align
	align_corners_ = param.align_corners();	
	
	// help
	vector<int> help_shape = {dst_h_,1};
	dst_h_temp_.Reshape(help_shape);
	h_temp_.Reshape(help_shape);
	Dtype* dst_h_temp_data = dst_h_temp_.mutable_cpu_data();
	for(int i=0; i<dst_h_; i++){
	    dst_h_temp_data[i] = i;
	}

	help_shape[0] = 1;
	help_shape[1] = dst_w_;
	dst_w_temp_.Reshape(help_shape);
	w_temp_.Reshape(help_shape);
	Dtype* dst_w_temp_data = dst_w_temp_.mutable_cpu_data();
	for(int i=0; i<dst_w_; i++){
	    dst_w_temp_data[i] = i;
	}

	// expand help mutrix
	expand_h_help_.Reshape(help_shape);
	caffe_set(expand_h_help_.count(), Dtype(1.), expand_h_help_.mutable_cpu_data());
	help_shape[0] = dst_h_;
	help_shape[1] = 1;
	expand_w_help_.Reshape(help_shape);
	caffe_set(expand_w_help_.count(), Dtype(1.), expand_w_help_.mutable_cpu_data());

	// init h,w
	help_shape[1] = dst_w_;
	h_.Reshape(help_shape);
	w_.Reshape(help_shape);
	h0_.Reshape(help_shape);
	w0_.Reshape(help_shape);
	h1_.Reshape(help_shape);
	w1_.Reshape(help_shape);
    }
    
    template <typename Dtype>
    void BilinearInterpolateLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
	// top
        vector<int> out_shape = {bottom[0]->num(),bottom[0]->channels(), dst_h_, dst_w_};
        top[0]->Reshape(out_shape);
    }
    
    template <typename Dtype>
    void BilinearInterpolateLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
        // straight copy
	if(dst_h_ == bottom[0]->height() && dst_w_ == bottom[0]->width()){
	    caffe_copy(bottom[0]->count(), bottom[0]->cpu_data(), top[0]->mutable_cpu_data());
	}

	// compute height and width of dst project to height and width of src
	int src_h = bottom[0]->height();
	int src_w = bottom[0]->width();

	Dtype* h_temp_data = h_temp_.mutable_cpu_data();
	const Dtype* dst_h_temp_data = dst_h_temp_.cpu_data();
	for(int i=0; i<dst_h_; i++){
	    if(!align_corners_){
	        h_temp_data[i] = float(src_h) / dst_h_ * (dst_h_temp_data[i] + 0.5) - 0.5;
	    }else{
	        h_temp_data[i] = float(src_h - 1) / (dst_h_ - 1) * dst_h_temp_data[i];
	    }

	    h_temp_data[i] = h_temp_data[i] > src_h - 1? src_h -1 : h_temp_data[i];
	    h_temp_data[i] = h_temp_data[i] < 0? 0 : h_temp_data[i];
	}

	Dtype* w_temp_data = w_temp_.mutable_cpu_data();
	const Dtype* dst_w_temp_data = dst_w_temp_.cpu_data();
	for(int i=0; i<dst_w_; i++){
	    if(!align_corners_){
	        w_temp_data[i] = float(src_w) / dst_w_ * (dst_w_temp_data[i] + 0.5) - 0.5;
	    }else{
	        w_temp_data[i] = float(src_w - 1) / (dst_w_ - 1) * dst_w_temp_data[i];
	    }
	    
	    w_temp_data[i] = w_temp_data[i] > src_w - 1? src_w -1 : w_temp_data[i];
	    w_temp_data[i] = w_temp_data[i] < 0? 0 : w_temp_data[i];
	}

	// expand dim
	// expand h
        caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, dst_h_, dst_w_, 1, Dtype(1.0),
			h_temp_.cpu_data(), expand_h_help_.cpu_data(), Dtype(0.),
			h_.mutable_cpu_data());	
	//expand w
	caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, dst_h_, dst_w_, 1, Dtype(1.0),
			expand_w_help_.cpu_data(), w_temp_.cpu_data(), Dtype(0.),
			w_.mutable_cpu_data());

	// compute four point
	const Dtype* h_data = h_.cpu_data();
	Dtype* h0_data = h0_.mutable_cpu_data();
	Dtype* h1_data = h1_.mutable_cpu_data();
	for(int i=0; i<h_.count(); i++){
	    int temp = floor(h_data[i]);
	    temp  = temp < 0? 0 : temp;
	    temp = temp > src_h - 2? src_h -2 : temp;
            h0_data[i] = temp;
	    h1_data[i] = temp + 1;
	}

	const Dtype* w_data = w_.cpu_data();
	Dtype* w0_data = w0_.mutable_cpu_data();
	Dtype* w1_data = w1_.mutable_cpu_data();
	for(int i=0; i<w_.count(); i++){
	    int temp = floor(w_data[i]);
	    temp  = temp < 0? 0 : temp;
	    temp = temp > src_w - 2? src_w -2 : temp;
            w0_data[i] = temp;
	    w1_data[i] = temp + 1;
	}
   
	// compute result
	int N = bottom[0]->num();
	int C = bottom[0]->channels();
	int H = dst_h_;
	int W = dst_w_;
	const Dtype* bottom_data = bottom[0]->cpu_data();
	Dtype* top_data = top[0]->mutable_cpu_data();
	for(int n=0; n<N; n++){
	    for(int c=0; c<C; c++){
	        for(int h=0; h<H; h++){
                for(int w=0; w<W; w++){
                    int top_index = n*(C*H*W) + c*(H*W) + h*W + w;

                    int h0_index = h0_data[h*W + w];
                    int w0_index = w0_data[h*W + w];
                    int h1_index = h1_data[h*W + w];
                    int w1_index = w1_data[h*W + w];
                    Dtype h_index = h_data[h*W + w];
                    Dtype w_index = w_data[h*W + w];

                    int q00_index = n*(C*src_h*src_w) + c*(src_h*src_w) + h0_index*src_w + w0_index;
                    int q01_index = n*(C*src_h*src_w) + c*(src_h*src_w) + h0_index*src_w + w1_index;
                    int q10_index = n*(C*src_h*src_w) + c*(src_h*src_w) + h1_index*src_w + w0_index;
                    int q11_index = n*(C*src_h*src_w) + c*(src_h*src_w) + h1_index*src_w + w1_index;

                    Dtype q00 = bottom_data[q00_index];
                    Dtype q01 = bottom_data[q01_index];
                    Dtype q10 = bottom_data[q10_index];
                    Dtype q11 = bottom_data[q11_index];

                    Dtype r0 = (w1_index - w_index)*q00 + (w_index - w0_index)*q01;
                    Dtype r1 = (w1_index - w_index)*q10 + (w_index - w0_index)*q11;
                    Dtype dst = (h1_index - h_index)*r0 + (h_index - h0_index)*r1;
                    top_data[top_index] = dst;
                }
		    }
	    }
	}

    }
    
    template <typename Dtype>
    void BilinearInterpolateLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
    }
    
    #ifdef CPU_ONLY
    //STUB_GPU(BilinearInterpolateLayer);
    #endif
    
    INSTANTIATE_CLASS(BilinearInterpolateLayer);
    REGISTER_LAYER_CLASS(BilinearInterpolate);
}  // namespace caffe
