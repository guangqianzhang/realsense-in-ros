#pragma once
#include <iomanip>
#include <librealsense2/rs.hpp>
#include "realsense/example-utils.hpp"
#include "realsense/cv-helpers.hpp"
#include "realsense/example.hpp"

namespace RealSense
{
    using namespace cv;
    using namespace std;
    using namespace rs2;
    vector<vector<Point>> contours; // 存储发现的轮廓对象
    vector<Vec4i> hierarchy;        // 图像拓扑结构
    Mat frame2Mat(frame &frames)
    {
        const int w = frames.as<video_frame>().get_width();
        const int h = frames.as<video_frame>().get_height();
        Mat color_image(Size(w, h), CV_8UC3, (void *)frames.get_data(), Mat::AUTO_STEP); // frame to Mat

        return color_image;
    }
    // 将表示距离的数字转为字符串的函数
    string Convert(float Num)
    {
        ostringstream oss;
        oss << Num;
        string str(oss.str());
        return str;
    }
    // 图像处理的函数
    Mat Morphological_process(Mat img)
    {
        Mat img_gray, dst;
        GaussianBlur(img, img, Size(7, 7), 7, 7);
        cvtColor(img, img_gray, COLOR_BGR2GRAY);
        threshold(img_gray, dst, 127, 255, THRESH_BINARY);
        return dst;
    }
    // 一个很多事很忙的函数
    Mat get_draw_contours(Mat dst_contour, Mat start_contour, depth_frame dep, int area)
    {
        /* 1.contours是一个双重向量，向量
               内每个元素保存了一组由连续的Point点构成的点的集合的向量，每一组Point点集就是一个轮廓。
               有多少轮廓，向量contours就有多少元素。
        2. hierarchy:Vec4i是Vec<int,4>的别名，定义了一个“向量内每一个元素包含了4个int型变量”的向量
         hierarchy向量内每一个元素的4个int型变量——hierarchy[i][0] ~hierarchy[i][3]，
         分别表示第i个轮廓的后一个轮廓、前一个轮廓、父轮廓、内嵌轮廓的索引编号
        3. CV_RETR_EXTERNAL只检测最外围轮廓，包含在外围轮廓内的内围轮廓被忽略
        CV_RETR_TREE， 检测所有轮廓，所有轮廓建立一个等级树结构。*/
        cv::findContours(dst_contour, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE, Point(0, 0));
        Scalar color = Scalar(100, 100, 200);
        vector<Moments> contour_moment(contours.size());
        vector<Point2f> centerpos(contours.size());
        vector<RotatedRect> rotated_rect(contours.size());   //定义带旋转角度的最小矩形集合
        Point2f p[4];       //四个角的坐标集合
        for (auto i = 0; i < contours.size(); ++i)
        {
            if (contourArea(contours[i]) > area)
            {
                cv::drawContours(start_contour, contours, i, color, 3, 8, hierarchy, 1, Point(0, 0));
                // 计算中心点坐标
                contour_moment[i] = moments(contours[i]); //计算图像距
                centerpos[i].x = contour_moment[i].m10 / contour_moment[i].m00;
                centerpos[i].y = contour_moment[i].m01 / contour_moment[i].m00;
                
                cv::circle(start_contour, centerpos[i], 3, Scalar(250, 250, 10));      // 绘画中心点
                float distance = dep.get_distance(centerpos[i].x, centerpos[i].y); // 距离单位为米
                
                
                if (distance >= 1.0)
                {
                    string ch = Convert(distance);
                    putText(start_contour, ch, centerpos[i], FONT_HERSHEY_SIMPLEX, 2, Scalar(100, 150, 250), 1, 8, false);
                    rotated_rect[i] = minAreaRect(contours[i]); //获取最小矩形集合
                    rotated_rect[i].points(p);                  //返回四个角的列表

                    for (int k = 0; k < 4; k++)
                    {
                        cv::line(start_contour, p[k], p[(k + 1) % 4], Scalar(0, 200, 200), 2, 8); //（k+1）%4防止数组越界
                    }
                }
            }

            
        }
        return start_contour;
    }
}