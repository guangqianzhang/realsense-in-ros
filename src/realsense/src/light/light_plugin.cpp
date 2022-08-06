#include <pluginlib/class_list_macros.h>
#include <ros/ros.h>
#include <nodelet/nodelet.h>
#include <cv_bridge/cv_bridge.h>
#include <image_transport/image_transport.h>
#include <sensor_msgs/Image.h>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>

#include "../cvfunction.hpp"
#include "traffic_light.hpp"
#include <rockauto_msgs/ImageObj.h>
#include <rockauto_msgs/ImageRect.h>
#include <rockauto_msgs/TrafficLightResult.h>

namespace Realsense
{
    using namespace cv;
    using namespace std;

    class light_plugin : public nodelet::Nodelet
    {
    private:
        std::string __NODE_NAME__;
        volatile bool light_running_;
        int binary_threshold_;
        int imgHeight_;
        int imgWidth_;

        int LightType = 0;
        

        boost::shared_ptr<traffic_light> light_;

        std::string light_topic_out_;
        std::string light_topic_in_;

        ros::Subscriber tracker_img_sub_;
        ros::Publisher tracker_img_pub_;

    public:
        light_plugin();
        ~light_plugin();
        virtual void onInit()
        {
            ROS_INFO("traffic light in onInit!");

            ros::NodeHandle private_nh = getPrivateNodeHandle();
            private_nh.param<std::string>("light_topic_in", light_topic_in_, "/light_in");
            private_nh.param<std::string>("light_topic_out", light_topic_out_, "/light_out");
            private_nh.param<int>("binary_threshold", binary_threshold_, 1);
            image_transport::ImageTransport it(private_nh);
            tracker_img_sub_ = private_nh.subscribe<rockauto_msgs::ImageObj>(light_topic_in_, 1, &light_plugin::image_cb, this);
            tracker_img_pub_ = private_nh.advertise<rockauto_msgs::TrafficLightResult>(light_topic_out_, 1);
        }
        void image_cb(const rockauto_msgs::ImageObjConstPtr &obj)
        {
            ROS_INFO("get data!");
            auto start = std::chrono::steady_clock::now();
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
            cv::Mat raw_img = (cv_ptr->image);
            cv::Mat workimage;

            if (imgcvRectArray.size())
            {

                for (size_t i = 0; i < imgRectArray.size(); i++)
                {
                    if (imgLabelArray[i].compare("traffic light") == 0)
                    {
                        rockauto_msgs::TrafficLightResultPtr light_result;
                        light_result->distance=ingDistanceArray[i];
                        
                        cv::Mat workimage = raw_img(imgcvRectArray[i]);
                        imgHeight_ = workimage.rows;
                        imgWidth_ = workimage.cols;
                        light_.reset(new traffic_light);
                        light_->setMask(workimage);
                        light_->ShowMask();
                        LightType = light_->count_color();
                        light_result->recognition_result=LightType;
                        
                        auto end = std::chrono::steady_clock::now();
                        std::chrono::duration<double> elapsed_seconds = end - start;
                        std::cout << "get light elapsed time: " << elapsed_seconds.count() << "s\n";
                    }
 
                }
            }
        }
    };
    light_plugin::light_plugin()
    {
        ROS_INFO("traffic light !");
        light_running_ = true;
    }
    light_plugin::~light_plugin()
    {
        ROS_INFO("bye Light !");
    }

    PLUGINLIB_EXPORT_CLASS(Realsense::light_plugin, nodelet::Nodelet)
}