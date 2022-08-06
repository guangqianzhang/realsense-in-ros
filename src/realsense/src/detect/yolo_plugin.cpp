#include <pluginlib/class_list_macros.h>
#include <ros/ros.h>
#include <nodelet/nodelet.h>
#include "../Yolo_plug.hpp"
#include "yolov7.hpp"
#include "../realsense.hpp"

#include <cv_bridge/cv_bridge.h>
#include <image_transport/image_transport.h>
#include <sensor_msgs/Image.h>
#include <opencv2/opencv.hpp>
namespace Realsense
{
    class Yolo_plugin : public nodelet::Nodelet
    {
    private:
        std::string __NODE_NAME__;
        volatile bool Yolo_running_;
        boost::shared_ptr<Yolo> yolo_;
        boost::shared_ptr<YOLOV7> yolov7_;

        std::string camera_topic_out_;
        std::string camera_topic_in_;

        image_transport::Subscriber camera_raw_sub_;
        image_transport::Publisher vision_detect_img_pub_;

        virtual void onInit()
        {
            Yolo_running_ = true;
            yolo_.reset(new Yolo);
            ros::NodeHandle private_nh = getPrivateNodeHandle();
            // ros::Rate loop_rate(10);
            private_nh.param<std::string>("camera_topic_in", camera_topic_in_, "/image_in");
            private_nh.param<std::string>("camera_topic_out", camera_topic_out_, "/image_out");
            image_transport::ImageTransport it(private_nh);
            // camera_raw_sub_ = it.subscribe(camera_topic_in_, 1, &Yolo_plugin::image_cb, this);
            vision_detect_img_pub_ = it.advertise(camera_topic_out_, 1);
            while (ros::ok())
            {
                /* code */
            }
            
            
        }
        void image_cb(const sensor_msgs::Image::ConstPtr &img)
        {
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
            ros::Time start_time = cv_ptr->header.stamp;
            std::vector<YoloDetSt> yoloRet;
            cv::Mat current_frame;
            current_frame = yolo_->detect(raw_img, yoloRet);
            
            
            ros::Time end_time = ros::Time::now();

            double fps = cv::getTickFrequency() / (end_time.nsec - start_time.nsec);
            std::cout << "FPS in nodelet: " << fps << std::endl; //

            cv_ptr->header.stamp = end_time;
            cv_ptr->encoding = "bgr8";
            cv_ptr->image = current_frame;
            vision_detect_img_pub_.publish(cv_ptr->toImageMsg());
        }

    public:
        Yolo_plugin(/* args */);
        ~Yolo_plugin();
    };
    Yolo_plugin::Yolo_plugin(/* args */)
    {
        ROS_INFO("SampleNodeletClass Constructor");
        __NODE_NAME__ = ros::this_node::getName();
    }
    Yolo_plugin::~Yolo_plugin()
    {
        ROS_INFO("SampleNodeletClass Destructor");
    }
    PLUGINLIB_EXPORT_CLASS(Realsense::Yolo_plugin, nodelet::Nodelet)
}
