#pragma once
#include <iostream>
#include <string>
#include <vector>
#include <opencv2/opencv.hpp>
using namespace std;
using namespace cv;

namespace mycv{
    class Tracker{
        public:
        Tracker(){
            cout<<"Go Tracker::Tracker!!"<<endl;

        }
        virtual ~Tracker(){
            cout<<"Go Tracker::~Tracker!!"<<endl;

        }
        //virtual 函数被继承，执行子lei
       virtual bool init(const Mat& initFrame,const std::vector<Rect>& initBoundingBoxs,
                                std::vector<float> ingDistanceArray,std::vector<std::string> imgLabelArray){
            cout<<"Go Tracker::init!!"<<endl;
        return false;
        }
       virtual bool track(const Mat& currentFrame, std::vector<Rect>& currentBoundingBox,const std::vector<Rect> &initBoundingBoxs,
                  std::vector<float> ingDistanceArray, std::vector<std::string> imgLabelArray){
            cout<<"Go Tracker::track!!"<<endl;
        return false;

        }
       virtual bool update(Rect& searchBox){
            cout<<"Go update!!"<<endl;

        return false;
        }

    };
}
