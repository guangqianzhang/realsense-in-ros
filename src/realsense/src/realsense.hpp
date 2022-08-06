#pragma once
#include <iomanip>
#include <librealsense2/rs.hpp>
#include "realsense/example-utils.hpp"
#include "realsense/cv-helpers.hpp"
#include "realsense/example.hpp"
#include "cvfunction.hpp"
#include "Yolo_plug.hpp"

#include <ros/ros.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/Image.h>
#include <image_transport/image_transport.h>

#include <rockauto_msgs/DetectedObjectArray.h>
#include <rockauto_msgs/ImageObj.h>
namespace RealSense
{
    using namespace rs2;
    using namespace cv;
    class realsense
    {
    private:
        const size_t inWidth = 300;
        const size_t inHeight = 300;
        const float WHRatio = inWidth / (float)inHeight;
        const float inScaleFactor = 0.007843f;
        const float meanVal = 127.5;

        boost::shared_ptr<Yolo> yolo_;
        std::string __NODE_NAME__;
        std::string realsense_colorMat_T_;
        std::string realsense_ImageObj_T_;
        std::string realsense_Light_T;

        image_transport::Subscriber camera_raw_sub_;
        image_transport::Publisher vision_detect_img_pub_;
        ros::Publisher vision_detect_msg_pub_;
        ros::Publisher light_detect_msg_pub_;

        bool detection_flag_ = true;
        bool Stop_Sign = false;

    public:
        realsense(ros::NodeHandle nh);
        ~realsense();
        void detectByOpencvGL();
        int captrue()
        {
            rs2::log_to_console(RS2_LOG_SEVERITY_ERROR);
            // Create a simple OpenGL window for rendering:
            window app(1280, 720, "RealSense Capture Example");
            // Declare depth colorizer for pretty visualization of depth data
            rs2::colorizer color_map;
            // Declare rates printer for showing streaming rates of the enabled streams.
            rs2::rates_printer printer;

            // Declare RealSense pipeline, encapsulating the actual device and sensors
            rs2::pipeline pipe;
            // Start streaming with default recommended configuration
            // The default video configuration contains Depth and Color streams
            // If a device is capable to stream IMU data, both Gyro and Accelerometer are enabled by default
            pipe.start();

            while (app) // Application still alive?
            {
                rs2::frameset data = pipe.wait_for_frames(). // Wait for next set of frames from the camera
                                     apply_filter(printer)
                                         .                    // Print each enabled stream frame rate
                                     apply_filter(color_map); // Find and colorize the depth data

                // The show method, when applied on frameset, break it to frames and upload each frame into a gl textures
                // Each texture is displayed on different viewport according to it's stream unique id
                app.show(data);
            }

            return EXIT_SUCCESS;
        }
        void Run()
        {
            // window app(1280, 720, "RealSense Capture Example");
            rs2::colorizer color_map;
            rs2::rates_printer printer;
            rs2::pipeline pipe;
            pipe.start();
            const auto window_name = "Display color_Image";
            namedWindow(window_name, WINDOW_AUTOSIZE);
            while (waitKey(1) < 0 && getWindowProperty(window_name, WND_PROP_AUTOSIZE) >= 0)
            {
                rs2::frameset data = pipe.wait_for_frames();
                // rs2::frameset data=pipe.wait_for_frames().apply_filter(printer).apply_filter(color_map);
                frame frames = data.get_color_frame().apply_filter(color_map); // search for depth date
                depth_frame depth = data.get_depth_frame();

                Mat color_image = frame2Mat(frames);
                detectByHull(color_image, depth, window_name);
            }
        }
        void detectByHull(Mat &color_image, const depth_frame &depth, string window_name)
        {
            Mat start_image = color_image;
            color_image = Morphological_process(color_image);
            cout << color_image.size() << endl;
            Mat color_image2 = get_draw_contours(color_image, start_image, depth, 5000);
            // app.show(depth);
            cout << color_image2.size() << endl;

            imshow(window_name, color_image);
            imshow("Display color_Image2", color_image2);
        }
        void detectByDnn()
        {
            // Start streaming from Intel RealSense Camera
            pipeline pipe;
            auto config = pipe.start();
            auto profile = config.get_stream(RS2_STREAM_COLOR)
                               .as<video_stream_profile>();
            rs2::align align_to(RS2_STREAM_COLOR);
            /*             // shape the image size to w=h
                        Size cropSize;
                        if (profile.width() / (float)profile.height() > WHRatio)
                        {
                            cropSize = Size(static_cast<int>(profile.height() * WHRatio),
                                            profile.height());
                        }
                        else
                        {
                            cropSize = Size(profile.width(),
                                            static_cast<int>(profile.width() / WHRatio));
                        }
                        // shape rect
                        Rect crop(Point((profile.width() - cropSize.width) / 2,
                                        (profile.height() - cropSize.height) / 2),
                                  cropSize); */

            const auto window_name = "Display Image";
            namedWindow(window_name, WINDOW_AUTOSIZE);
            yolo_.reset(new Yolo);

            while (getWindowProperty(window_name, WND_PROP_AUTOSIZE) >= 0 && ros::ok())
            {

                // Wait for the next set of frames
                auto data = pipe.wait_for_frames();
                double start_time = (double)cv::getTickCount();
                ros::Time time = ros::Time::now();
                // Make sure the frames are spatially aligned
                data = align_to.process(data);

                auto color_frame = data.get_color_frame();
                auto depth_frame = data.get_depth_frame();
                // If we only received new depth frame,
                // but the color did not update, continue
                static int last_frame_number = 0;
                if (color_frame.get_frame_number() == last_frame_number)
                    continue;
                last_frame_number = static_cast<int>(color_frame.get_frame_number());

                // Convert RealSense frame to OpenCV matrix:
                auto color_mat = frame_to_mat(color_frame);
                auto depth_mat = depth_frame_to_meters(depth_frame);
                Rect depth_mat_rect(Point(0, 0), Point(depth_mat.cols, depth_mat.rows));
                // // Crop both color and depth frames//裁减
                // color_mat = color_mat(crop);
                // depth_mat = depth_mat(crop);
                cv::Mat work_frame;
                color_mat.copyTo(work_frame);
                cv_bridge::CvImagePtr cv_ptr(new cv_bridge::CvImage);

                // detection
                if (detection_flag_)
                {

                    std::vector<YoloDetSt> yoloRet;
                    std::vector<YoloDetSt> detectRet;

                    yolo_->detect(work_frame, yoloRet);
                    //用于发送的消息
                    rockauto_msgs::ImageObjPtr ImageObj(new rockauto_msgs::ImageObj);
                    cv_ptr->header.stamp = time;
                    cv_ptr->encoding = "bgr8";
                    cv_ptr->image = color_mat;
                    ImageObj->roi_image = *cv_ptr->toImageMsg();

                    std::cout << "yoloRet.size:" << yoloRet.size() << std::endl;

                    for (size_t i = 0; i < yoloRet.size(); i++)
                    {

                        // YoloDetSt detectSt;
                        bool is_rect =
                            (0 <= yoloRet[i].rect.x && 0 <= yoloRet[i].rect.width &&
                             yoloRet[i].rect.x + yoloRet[i].rect.width <= depth_mat.cols &&
                             0 <= yoloRet[i].rect.y && 0 <= yoloRet[i].rect.height &&
                             yoloRet[i].rect.y + yoloRet[i].rect.height <= depth_mat.rows);
                        if (!is_rect)
                        {
                            // 不合法，此时continue、break或者return.
                            // continue;
                            cout << "rect:" << yoloRet[i].rect << endl;
                            yoloRet[i].rect = yoloRet[i].rect & depth_mat_rect;
                            cout << "rect:" << yoloRet[i].rect << endl;
                        }
                        rockauto_msgs::ImageRect ObjRect;
                        ObjRect.x = yoloRet[i].rect.x;
                        ObjRect.y = yoloRet[i].rect.y;
                        ObjRect.width = yoloRet[i].rect.width;
                        ObjRect.height = yoloRet[i].rect.height;
                        ObjRect.score = yoloRet[i].confidences;
                        ImageObj->obj.push_back(ObjRect);

                        ImageObj->type.push_back(yoloRet[i].label);

                        /*                     cout << "num_obj:" << i << endl;
                                            cout << "rect:" << yoloRet[i].rect << endl;
                                            // color_mat = color_mat(it->rect);
                                            cout << "depMat shape:" << depth_mat.size() << endl; */
                        // resize the roi rect*2/4
                        Rect depth_roi;

                        if (yoloRet[i].label.compare("stop sign") == 0)
                        {
                            Stop_Sign = true;
                            ROS_INFO("get stop sign!!");
                        }
                        cv::Size dep_roi;
                        int width = yoloRet[i].rect.width / 2;
                        int height = yoloRet[i].rect.height / 2;
                        if ((width) > (height))
                        {
                            dep_roi = cv::Size(-height, -height);
                        }
                        else
                        {
                            dep_roi = cv::Size(-width, -width);
                        }
                        depth_roi = rectCenterScale(yoloRet[i].rect, dep_roi);

                        auto depth_mat_Roi = depth_mat(depth_roi);
                        // cv::imshow("roi", depth_mat_Roi);
                        Scalar m = mean(depth_mat_Roi);
                        float distance = m[0];
                        ImageObj->distanse.push_back(distance);
                        yoloRet[i].rect = depth_roi;
                        std::ostringstream ss;
                        ss << yoloRet[i].label << ":" << m[0] << "m";
                        yoloRet[i].label = ss.str();
                        cout << "lable:" << yoloRet[i].label << endl;

                    } // end  for

                    vision_detect_msg_pub_.publish(*ImageObj);

                    yolo_->drowBoxes(work_frame, yoloRet);
                } // end if detection_flag_
                double end_time = (double)cv::getTickCount();
                double fps = cv::getTickFrequency() / (end_time - start_time);
                ROS_INFO("FPS in app: %0.3f", fps); //

                cv::imshow(window_name, work_frame);
                // cv::imshow("depth mat", depth_mat);
                if (waitKey(1) >= 0)
                    break;
            }
        }
    };

    realsense::realsense(ros::NodeHandle nh)
    {
        __NODE_NAME__ = ros::this_node::getName();
        ros::NodeHandle private_nh = nh;
        private_nh.param<std::string>("camera_topic", realsense_colorMat_T_, "/image_raw");
        std::cout << __NODE_NAME__ << ":camera_topic:" << realsense_colorMat_T_.c_str() << std::endl;
        private_nh.param<std::string>("ImageObj_topic", realsense_ImageObj_T_, "/image_Obj");
        std::cout << __NODE_NAME__ << ":ImageObj_topic:" << realsense_ImageObj_T_.c_str() << std::endl;
        private_nh.param<std::string>("Light_topic", realsense_Light_T, "/light_Obj");
        std::cout << __NODE_NAME__ << ":Light_topic:" << realsense_Light_T.c_str() << std::endl;

        private_nh.param<bool>("detection_flag", detection_flag_, "true");
        std::cout << __NODE_NAME__ << ":detection_flag:" << detection_flag_ << std::endl;

        image_transport::ImageTransport it(private_nh);
        vision_detect_img_pub_ = it.advertise(realsense_colorMat_T_, 1);
        vision_detect_msg_pub_ = private_nh.advertise<rockauto_msgs::ImageObj>(realsense_ImageObj_T_, 1);
        light_detect_msg_pub_ = private_nh.advertise<rockauto_msgs::ImageObj>(realsense_Light_T, 1);
    }

    realsense::~realsense()
    {
    }

    void realsense::detectByOpencvGL()
    {
    }
}