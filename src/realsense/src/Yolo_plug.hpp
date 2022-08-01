#pragma once
#include <iostream>
#include "opencv2/dnn.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/opencv.hpp"
#include <time.h>
#include <ctime>
#include <ros/ros.h>
#include <cv_bridge/cv_bridge.h>
#include <image_transport/image_transport.h>
#include <sensor_msgs/Image.h>
#include <rockauto_msgs/DetectedObjectArray.h>
#include <rockauto_msgs/ImageObj.h>
typedef struct YoloDetSt
{
	std::string label;
	float confidences;
	// 	 　　int x; /* 方形的左上角的x-坐标 */
	// 　　int y; /* 方形的左上角的y-坐标*/
	// 　　int width; /* 宽 */
	// 　　int height; /* 高 */
	cv::Rect rect;
} YoloDetSt;

class Yolo
{
public:
	Yolo();
	~Yolo();
	int loadModel();
	cv::Mat detect(cv::Mat &img, std::vector<YoloDetSt> &yoloRet);
	void drowBoxes(cv::Mat &img, std::vector<YoloDetSt> &yoloRet);

private:
	std::string medel_path;
	std::string yoloModelWeights;
	std::string yoloModelConfiguration;
	std::string yoloclassfile;
	std::vector<std::string> m_outNames;
	bool m_isGpu = false;
	std::vector<std::string> m_classes;
	cv::dnn::Net m_net;
bool HAVE_INSTRINSIS=false;
	// Yolo��������
	float m_confThreshold = 0.5;
	float m_nmsThreshold = 0.4;
	float m_scale = 0.00392;
	cv::Scalar m_mean = {0, 0, 0};
	bool m_swapRB = true;
	int m_inpWidth = 416;
	int m_inpHeight = 416;

	// ����ͼƬ�������Ƶ�Ĳ���
	int m_saveH = 0;
	int m_saveW = 0;
	cv::VideoWriter m_viderWriter;
	std::string m_viderName;
	int m_frames = 0;

	cv::Size image_size_;
	cv::Mat camera_instrinsics_;
	cv::Mat distortion_coefficients_;
	cv::Mat CameraExtrinsicMat_;
	cv::Mat current_frame_;
	void initParams();
	int runningYolo(cv::Mat &img, std::vector<YoloDetSt> &yoloRet);
	void postprocess(cv::Mat &frame, const std::vector<cv::Mat> &outs, cv::dnn::Net &net, std::vector<YoloDetSt> &yoloRet);
};
#include <fstream>

#include "glog/logging.h"
#include <yaml-cpp/yaml.h>

Yolo::Yolo()
{
	initParams();
}

Yolo::~Yolo()
{
	m_viderWriter.release();
}
void Yolo::initParams()
{

	// modelconfig
	std::string config_file_path = "/home/zgq/Documents/depth_camera/catkin_RealSense/src/cfg/modelconfig.yaml";
	YAML::Node config = YAML::LoadFile(config_file_path);
	medel_path = config["medel_path"].as<std::string>();
	yoloclassfile = medel_path + config["yoloclassfile"].as<std::string>();

	yoloModelConfiguration = medel_path + config["yoloModelConfiguration"].as<std::string>();
	yoloModelWeights = medel_path + config["yoloModelWeights"].as<std::string>();
	// private_node_handle.param<bool>("isGpu", m_isGpu, true);
	m_isGpu=config["is_gpu"].as<bool>();
	printf("use gpu:%d\n",m_isGpu);
	// m_isGpu = true;
	// LOG(INFO) << __APP_NAME__ << ":isGpu:" << m_isGpu;
	// LOG(INFO) << __APP_NAME__ << ":medel_path:" << medel_path.c_str();
	// LOG(INFO) << __APP_NAME__ << ":yoloclassfile:" << yoloclassfile.c_str();
	// LOG(INFO) << __APP_NAME__ << ":yoloModelConfiguration:" << yoloModelConfiguration.c_str();
	// LOG(INFO) << __APP_NAME__ << ":yoloModelWeights:" << yoloModelWeights.c_str();

	// std::string config_file_Camera_path = "/home/brt/Documents/catktin_brt/src/vision_camera/cfg/lidar_camera_calibration64.yaml";
	// // YAML::Node config_camera = YAML::LoadFile(config_file_Camera_path);
	// cv::FileStorage config_camera(config_file_Camera_path, cv::FileStorage::READ);

	// config_camera["CameraMat"] >> camera_instrinsics_;
	// config_camera["DistCoeff"] >> distortion_coefficients_;
	// config_camera["CameraExtrinsicMat"] >> CameraExtrinsicMat_;
	// HAVE_INSTRINSIS=true;

	loadModel();
}

int Yolo::loadModel()
{
	int backendId;
	int targetId;
	// cpu or gpu
	// std::cout<<"use GPU:"<<m_isGpu<<std::endl;
	if (m_isGpu)
	{
		backendId = cv::dnn::DNN_BACKEND_CUDA;
		targetId = cv::dnn::DNN_TARGET_CUDA;
	}
	else
	{
		backendId = cv::dnn::DNN_BACKEND_OPENCV;
		targetId = cv::dnn::DNN_TARGET_CPU;
	}

	// Open file with classes names.
	if (!yoloclassfile.empty())
	{
		std::ifstream ifs(yoloclassfile.c_str());
		if (!ifs.is_open())
		{
			std::string error = "File " + yoloclassfile + " not found";
			std::cout << error << std::endl;
			return -1;
		}
		std::string line;
		while (std::getline(ifs, line))
		{
			m_classes.push_back(line);
		}
	}

	// Load a model.
	m_net = cv::dnn::readNet(yoloModelWeights, yoloModelConfiguration);
	m_net.setPreferableBackend(backendId);
	m_net.setPreferableTarget(targetId);

	m_outNames = m_net.getUnconnectedOutLayersNames();

	return 0;
}

int Yolo::runningYolo(cv::Mat &img, std::vector<YoloDetSt> &yoloRet)
{
	// Create a 4D blob from a frame.
	cv::Mat blob;
	cv::Mat frame;
	cv::Size inpSize(m_inpWidth > 0 ? m_inpWidth : img.cols,
					 m_inpHeight > 0 ? m_inpHeight : img.rows);
	cv::dnn::blobFromImage(img, blob, m_scale, inpSize, m_mean, m_swapRB, false);

	// Run a model.
	m_net.setInput(blob);
	if (m_net.getLayer(0)->outputNameToIndex("im_info") != -1) // Faster-RCNN or R-FCN
	{
		cv::resize(img, img, inpSize);
		cv::Mat imInfo = (cv::Mat_<float>(1, 3) << inpSize.height, inpSize.width, 1.6f);
		m_net.setInput(imInfo, "im_info");
	}
	std::vector<cv::Mat> outs;
	m_net.forward(outs, m_outNames);
	postprocess(img, outs, m_net, yoloRet);
	return 0;
}

void Yolo::postprocess(cv::Mat &frame, const std::vector<cv::Mat> &outs, cv::dnn::Net &net, std::vector<YoloDetSt> &yoloRet)
{
	static std::vector<int> outLayers = net.getUnconnectedOutLayers();
	static std::string outLayerType = net.getLayer(outLayers[0])->type;

	std::vector<int> classIds;
	std::vector<float> confidences;
	std::vector<cv::Rect> boxes;
	if (net.getLayer(0)->outputNameToIndex("im_info") != -1)
	{
		// Network produces output blob with a shape 1x1xNx7 where N is a number of
		// detections and an every detection is a vector of values
		// [batchId, classId, confidence, left, top, right, bottom]
		CV_Assert(outs.size() == 1);
		float *data = (float *)outs[0].data;
		for (size_t i = 0; i < outs[0].total(); i += 7)
		{
			float confidence = data[i + 2];
			if (confidence > m_confThreshold)
			{
				int left = (int)data[i + 3];
				int top = (int)data[i + 4];
				int right = (int)data[i + 5];
				int bottom = (int)data[i + 6];
				int width = right - left + 1;
				int height = bottom - top + 1;
				classIds.push_back((int)(data[i + 1]) - 1); // Skip 0th background class id.
				boxes.push_back(cv::Rect(left, top, width, height));
				confidences.push_back(confidence);
			}
		}
	}
	else if (outLayerType == "DetectionOutput")
	{
		// Network produces output blob with a shape 1x1xNx7 where N is a number of
		// detections and an every detection is a vector of values
		// [batchId, classId, confidence, left, top, right, bottom]
		CV_Assert(outs.size() == 1);
		float *data = (float *)outs[0].data;
		for (size_t i = 0; i < outs[0].total(); i += 7)
		{
			float confidence = data[i + 2];
			if (confidence > m_confThreshold)
			{
				int left = (int)(data[i + 3] * frame.cols);
				int top = (int)(data[i + 4] * frame.rows);
				int right = (int)(data[i + 5] * frame.cols);
				int bottom = (int)(data[i + 6] * frame.rows);
				int width = right - left + 1;
				int height = bottom - top + 1;
				classIds.push_back((int)(data[i + 1]) - 1); // Skip 0th background class id.
				boxes.push_back(cv::Rect(left, top, width, height));
				confidences.push_back(confidence);
			}
		}
	}
	else if (outLayerType == "Region")
	{
		for (size_t i = 0; i < outs.size(); ++i)
		{
			// Network produces output blob with a shape NxC where N is a number of
			// detected objects and C is a number of classes + 4 where the first 4
			// numbers are [center_x, center_y, width, height]
			float *data = (float *)outs[i].data;
			for (int j = 0; j < outs[i].rows; ++j, data += outs[i].cols)
			{
				cv::Mat scores = outs[i].row(j).colRange(5, outs[i].cols);
				cv::Point classIdPoint;
				double confidence;
				minMaxLoc(scores, 0, &confidence, 0, &classIdPoint);
				if (confidence > m_confThreshold)
				{
					int centerX = (int)(data[0] * frame.cols);
					int centerY = (int)(data[1] * frame.rows);
					int width = (int)(data[2] * frame.cols);
					int height = (int)(data[3] * frame.rows);
					int left = centerX - width / 2;
					int top = centerY - height / 2;

					classIds.push_back(classIdPoint.x);
					confidences.push_back((float)confidence);
					boxes.push_back(cv::Rect(left, top, width, height));
				}
			}
		}
	}
	else
	{
		std::cout << "Unknown output layer type: " + outLayerType << std::endl;
	}

	std::vector<int> indices;
	YoloDetSt detSt;
	cv::dnn::NMSBoxes(boxes, confidences, m_confThreshold, m_nmsThreshold, indices);
	for (size_t i = 0; i < indices.size(); ++i)
	{
		int idx = indices[i];

		std::string label;
		if (!m_classes.empty())
		{
			CV_Assert(classIds[idx] < (int)m_classes.size());
		}
		yoloRet.push_back(YoloDetSt{m_classes[classIds[idx]], confidences[idx], boxes[idx]});
	}
}

void Yolo::drowBoxes(cv::Mat &img, std::vector<YoloDetSt> &yoloRet)
{
	for (__uint8_t i = 0; i < yoloRet.size(); i++)
	{
		cv::rectangle(img, yoloRet[i].rect, cv::Scalar(0, 0, 255));
		std::string label = cv::format("%.2f", yoloRet[i].confidences);
		label = yoloRet[i].label + ": " + label;
		int baseLine;
		cv::Size labelSize = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);

		int top = cv::max(yoloRet[i].rect.y, labelSize.height);
		rectangle(img, cv::Point(yoloRet[i].rect.x, top - labelSize.height),
				  cv::Point(yoloRet[i].rect.x + labelSize.width, top + baseLine), cv::Scalar(255, 255, 255), cv::FILLED);
		putText(img, label, cv::Point(yoloRet[i].rect.x, top), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar());
	}
}

cv::Mat Yolo::detect(cv::Mat &raw_img, std::vector<YoloDetSt> &yoloRet)
{
	double start_time = (double)cv::getTickCount();
	cv::Mat undistorted_image;
		if(HAVE_INSTRINSIS){

	cv::undistort(raw_img, current_frame_, camera_instrinsics_, distortion_coefficients_);
	}
	else{current_frame_=raw_img;}
	image_size_.height = current_frame_.rows;
	image_size_.width = current_frame_.cols;

	runningYolo(current_frame_, yoloRet);
	drowBoxes(current_frame_, yoloRet);

	double end_time = (double)cv::getTickCount();
	double fps = cv::getTickFrequency() / (end_time - start_time);
	double spend_time = (end_time - start_time) / cv::getTickFrequency();
	std::string FPS = "FPS:" + cv::format("%.2f", fps) + "  spend time:" + cv::format("%.2f", spend_time * 1000) + "ms";
	putText(current_frame_, FPS, cv::Point(0, 15), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 255), 1, 8, 0);
	// std::cout << "have done detect image! spend time" << spend_time * 1000 << "ms" << std::endl;
	return current_frame_;
}
