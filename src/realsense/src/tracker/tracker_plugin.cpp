#include <pluginlib/class_list_macros.h>
#include <ros/ros.h>
#include <nodelet/nodelet.h>
#include <cv_bridge/cv_bridge.h>
#include <image_transport/image_transport.h>
#include <sensor_msgs/Image.h>
#include <opencv2/opencv.hpp>
#include "tracker.hpp"
#include "MultiTemplateTracker.hpp"
#include <rockauto_msgs/ImageObj.h>
#include <rockauto_msgs/ImageRect.h>

namespace Realsense
{
    using namespace cv;
    using namespace std;

    class tracker_plugin : public nodelet::Nodelet
    {
    private:
        std::string __NODE_NAME__;
        volatile bool Tracker_running_;

        std::string tracker_topic_out_;
        std::string tracker_topic_in_;

        ros::Subscriber tracker_img_sub_;
        image_transport::Publisher tracker_img_pub_;

         mycv::MTTracker::Params mtparams = mycv::MTTracker::Params();
         Ptr<mycv::Tracker> tracker = new mycv::MTTracker(mtparams); //父类下城，调用字类

        struct global
        {
            bool getObjRct = false;
            bool isRoiReady = false;
            cv::Mat displayImg;

        } global; // strcut global
    public:
        tracker_plugin(/* args */);
        ~tracker_plugin();
        virtual void onInit()
        {
            Tracker_running_ = true;

            ros::NodeHandle private_nh = getPrivateNodeHandle();
            private_nh.param<std::string>("tracker_topic_in", tracker_topic_in_, "/tracker_in");
            private_nh.param<std::string>("tracker_topic_out", tracker_topic_out_, "/tracker_out");
            image_transport::ImageTransport it(private_nh);
            tracker_img_sub_ = private_nh.subscribe<rockauto_msgs::ImageObj>(tracker_topic_in_, 1, &tracker_plugin::image_cb, this);
            tracker_img_pub_ = it.advertise(tracker_topic_out_, 1);

           
            mtparams.expandWidth = 80;
            mtparams.sigma = Point2i(0.5, 0.5); //越大越均匀
            mtparams.numPoints = 800;
            mtparams.alpha = 0.7;
            
        }
        void image_cb(const rockauto_msgs::ImageObjConstPtr &obj)
        {
            sensor_msgs::Image img = obj->roi_image;
            std::vector<rockauto_msgs::ImageRect> imgRectArray = obj->obj;
            std::vector<std::string> imgLabelArray = obj->type;
            std::vector<float> ingDistanceArray = obj->distanse;

            std::vector<cv::Rect> imgcvRectArray;
            for (auto it = imgRectArray.begin(); it != imgRectArray.end(); ++it)
            {
                cv::Rect rect;
                rect.x = it->x;
                rect.y = it->y;
                rect.width = it->width;
                rect.height = it->height;
                imgcvRectArray.push_back(rect);
            }
            cv_bridge::CvImagePtr cv_ptr;
            try
            {
                cv_ptr = cv_bridge::toCvCopy(img, "bgr8");
            }
            catch (cv_bridge::Exception &e)
            {
                ROS_INFO("cv_bridge Copy failed!");
                return;
            }
            cv::Mat raw_img = cv_ptr->image;
            CV_Assert(!raw_img.empty());
            // raw_img.copyTo(global.displayImg);

            cv::Mat currentFrame;
            cvtColor(raw_img, currentFrame, CV_BGR2GRAY);
            
            tracker->init(currentFrame, imgcvRectArray, ingDistanceArray, imgLabelArray);
            // //开始跟踪
            std::vector<Rect> CurrentBoundingBoxs; //传出参数
            tracker->track(currentFrame, CurrentBoundingBoxs,imgcvRectArray, ingDistanceArray, imgLabelArray);
            // //显示当前跟踪结果
            cout<<"currentBoxs:"<<CurrentBoundingBoxs.size()<<endl;
            for (auto it = CurrentBoundingBoxs.begin(); it != CurrentBoundingBoxs.end(); ++it)
            {

                rectangle(raw_img, *it, Scalar(0, 0, 255), 2);
            }
            imshow("winName", raw_img);

            cv_ptr->image = raw_img;
            tracker_img_pub_.publish(cv_ptr->toImageMsg());
        }

    };

    tracker_plugin::tracker_plugin(/* args */)
    {
    }

    tracker_plugin::~tracker_plugin()
    {
    }

    PLUGINLIB_EXPORT_CLASS(Realsense::tracker_plugin, nodelet::Nodelet)
}