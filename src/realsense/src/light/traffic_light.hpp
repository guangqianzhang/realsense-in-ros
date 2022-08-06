#pragma once
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
using namespace cv;
using namespace std;
class traffic_light
{

private:
    //利用hsv模型提取image中的颜色
    cv::Mat mask_red;
    cv::Mat mask_yellow;
    cv::Mat mask_green;
    enum Traffic_Light
    {
        RED = 1,
        YELLOW = 2,
        GREEN = 3
    };
    Traffic_Light light;

private:
    void mask(const cv::Mat &image, double minH, double maxH, double minS, double maxS, cv::Mat &mask)
    {
        cv::Mat hsv;
        cv::Mat mask1;
        cv::Mat mask2;
        cv::cvtColor(image, hsv, CV_BGR2HSV);
        cv::Mat channels[3];
        cv::split(hsv, channels);
        cv::threshold(channels[0], mask1, maxH, 255, cv::THRESH_BINARY_INV);

        cv::threshold(channels[0], mask2, minH, 255, cv::THRESH_BINARY);
        cv::Mat hueMask;
        if (minH < maxH)
            hueMask = mask1 & mask2;

        cv::threshold(channels[1], mask1, maxS, 255, cv::THRESH_BINARY_INV);
        cv::threshold(channels[1], mask2, minS, 255, cv::THRESH_BINARY);
        cv::Mat satMask;
        satMask = mask1 & mask2;

        mask = hueMask & satMask;
    }

public:
    traffic_light(/* args */);
    ~traffic_light();
    Traffic_Light count_color()
    {
        int count_red = 0;
        int count_green = 0;
        int count_yellow = 0;
        //遍历像素点
        for (int i = 0; i < mask_red.rows; i++)
        {
            for (int j = 0; j < mask_red.cols; j++)
            {
                if (mask_red.at<uchar>(i, j) == 255)
                    count_red++;
            }
        }
        for (int i = 0; i < mask_yellow.rows; i++)
        {
            for (int j = 0; j < mask_yellow.cols; j++)
            {
                if (mask_yellow.at<uchar>(i, j) == 255)
                    count_yellow++;
            }
        }
        for (int i = 0; i < mask_green.rows; i++)
        {
            for (int j = 0; j < mask_green.cols; j++)
            {
                if (mask_green.at<uchar>(i, j) == 255)
                    count_green++;
            }
        }
        if ((count_red > count_yellow) && (count_red > count_green))
            light = Traffic_Light::RED;
        else if ((count_yellow > count_red) && (count_yellow > count_green))
            light = Traffic_Light::YELLOW;
        else if ((count_green > count_yellow) && (count_green > count_red))
            light = Traffic_Light::GREEN;
        return light;
    }
    void setMask(const cv::Mat image1)
    {
        mask(image1, 150, 180, 43, 255, mask_red);  //红色
        mask(image1, 11, 34, 43, 255, mask_yellow); //黄色
        mask(image1, 35, 100, 43, 255, mask_green); //绿色
    }
    void ShowMask()
    {
        cv::namedWindow("red");
        cv::imshow("red", mask_red);
        cv::namedWindow("yellow");
        cv::imshow("yellow", mask_yellow);
        cv::namedWindow("green");
        cv::imshow("green", mask_green);
        waitKey(10);
    }
};

traffic_light::traffic_light(/* args */)
{
}

traffic_light::~traffic_light()
{
}
