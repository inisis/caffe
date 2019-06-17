/***************************************************************************
*
* Copyright(c) 2015 Baidu.com, Inc. All Rights Reserved
*
**************************************************************************/

/**
* @file caffe_interface.h
* @author liuguoyi01(com@baidu.com)
* @date 2015 / 08/23 15:00:37
* @brief
*
**/
#ifndef  CAFFE_MASTER_CAFFE_INTERFACE_H
#define  CAFFE_MASTER_CAFFE_INTERFACE_H

#include <vector>
#include <string>
#include <opencv/cxcore.h>
#include <boost/shared_ptr.hpp>
    
#ifdef _WIN32
#ifdef CAFFE_EXPORTS
#define CAFFE_API __declspec(dllexport)
#else
#define CAFFE_API __declspec(dllimport)
#endif
#else
#define CAFFE_API __attribute__ ((visibility("default")))
#endif
    
namespace vis
{
    class CaffePreProcessParam{
    public:
        CaffePreProcessParam() : _resizewidth(0), _resizeheight(0), _scale(1.0) {}
        inline void set_input_dim(int item, int channel, int height, int width){
            _input_dim.resize(4);
            _input_dim[0] = item ; 
            _input_dim[1] = channel; 
            _input_dim[2] = height ; 
            _input_dim[3] = width ; 
        }
        inline void set_scale(float scale) {
            _scale = scale;
        }
        inline void set_meanvalues(int b, int g, int r){
            _mean_value.resize(3); 
            _mean_value[0] = b;
            _mean_value[1] = g;
            _mean_value[2] = r;
        }
        std::string _mean_file; //
        std::vector < int> _mean_value;
        int _resizewidth;
        int _resizeheight;
        float _scale;
        std::vector < int> _input_dim;
    };

    class CaffeParam{
    public:
        CaffeParam(){}
        std::string _models_file;
        std::string _weights_file;
    };

    //wrapper of Caffe::Blob
    class ICaffeBlob {
    public:
        virtual ~ICaffeBlob() {}
        virtual const std::vector < int>& shape() = 0;
        void reshape(const std::vector < int>& shape);
        virtual const float * cpu_data() const = 0;
        virtual void set_cpu_data(float * data) = 0;
        virtual float * mutable_cpu_data() = 0;
        virtual void savetofile(const std::string & filename) = 0;
        virtual bool getdata(std::vector < float> & outputdata, 
            std::vector<int> & outputshape) = 0;
        virtual int count() = 0;
    protected:
        ICaffeBlob(){};
    };

    class ICaffePreProcess{
    public:
        virtual ~ICaffePreProcess(){} 
        virtual ICaffePreProcess * clone() = 0;
        virtual boost::shared_ptr < ICaffeBlob> process(const cv::Mat & image) = 0;
    protected:
        ICaffePreProcess(){};
    };

    //wrapper std::vector < boost::shared_ptr < ICaffeBlob> > , to avoid alloc internal and release outside
    class ICaffeLayerDatas {
    public:
        virtual ~ICaffeLayerDatas(){};
        virtual void push_back(boost::shared_ptr < ICaffeBlob> & blob) = 0;
        virtual int getsize()const = 0;
        virtual boost::shared_ptr < ICaffeBlob>& getdata(int i)  = 0;
        virtual void clear() = 0;
    protected:
        ICaffeLayerDatas(){};
    };
    CAFFE_API ICaffeLayerDatas* createcaffelayerdatas();
    class ICaffePredict {
    public:
        virtual ~ICaffePredict() {}
        virtual ICaffePredict * clone() = 0;
        //some raw api
        virtual bool predict(const ICaffeLayerDatas * inputdatas, 
            const std::vector < std::string> & layernames,
            ICaffeLayerDatas * outputdatas) = 0;
        //make it a inline call so malloc and free of vector are both outside of so. 
        inline bool predict(const std::vector < boost::shared_ptr < ICaffeBlob> > & inputblobs, 
            const std::vector < std::string> & layernames,
            std::vector < boost::shared_ptr < ICaffeBlob> > &outputblobs){
            boost::shared_ptr < ICaffeLayerDatas> inputdatas(createcaffelayerdatas());
            boost::shared_ptr < ICaffeLayerDatas> outputdatas(createcaffelayerdatas());
            for (int i = 0;i < inputblobs.size(); i++)
            {
                inputdatas->push_back(*const_cast<boost::shared_ptr < ICaffeBlob>* >(&inputblobs[i]));
            }
            bool r = this->predict(inputdatas.get(), layernames, outputdatas.get());
            outputblobs.clear();
            if (r){
                for (int i = 0; i < outputdatas->getsize(); i++)
                {
                    outputblobs.push_back(outputdatas->getdata(i));
                }
            }
            return true;
        }
    protected:
        ICaffePredict() {}
    };



    CAFFE_API void initcaffeglobal(int argc, char** argv,int deviceid);
    CAFFE_API ICaffePredict * createcaffepredict(const CaffeParam &param);
    CAFFE_API ICaffePreProcess * createcaffepreprocess(const CaffePreProcessParam & param);
    CAFFE_API ICaffeBlob * createcaffeblob(const std::vector < int>& shape);

}

#endif  //CAFFE_MASTER_CAFFE_INTERFACE_H

/* vim: set expandtab ts=4 sw=4 sts=4 tw=100: */
