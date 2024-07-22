//Updated code to record timestamp whenever person enters or exits ROI:
//the timestamp function is declared in its own class. 
//This code uses the yolov10s.onnx model

#include "inference.h"
#include "Timestamp.h"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <unordered_set>
#include <fstream>
using namespace std;
using namespace cv;

//global variables
vector<vector<Point>> rois;
vector<Point> current_roi;
int roi_count = 0;
string current_time;
bool inside_ROI = false;
string roi_names[] = {"ROI 1", "ROI 2", "ROI 3"};
bool prev_state[] = { 0, 0, 0 };

struct Event {
    string timestamp;
    int eventType; // 1 for inside, 0 outside

    // Constructor to easily create events
    Event(const string& ts, int type) : timestamp(ts), eventType(type) {}
};
// Create a map to store events for different ROIs
map<string, vector<Event>> roiEvents;

Mat drawROIpolygons(const Mat& image, const vector<Point>& polygon, const Scalar& color) {
    Mat result = image.clone();
    polylines(result, polygon, true, color, 2); //draw polygon
    return result;
}

void writeEventsToCSV(const std::map<std::string, std::vector<Event>>& Events, const std::string& filename) {
    // Open a file in write mode
    ofstream file(filename);

    // Check if the file is open
    if (file.is_open()) {
        // Write the header
        file << "ROI,Timestamp,Event\n";

        // Write each event
        for (const auto& office : Events) {
            for (const auto& event : office.second) {
                file << office.first << "," << event.timestamp << "," << (event.eventType == 1 ? "Enter" : "Exit") << "\n";
            }
        }

        // Close the file
        file.close();
        cout << "Events successfully written to " << filename << endl;
    }
    else {
        cerr << "Unable to open file: " << filename << endl;
    }
}
// Define the classes to keep
const unordered_set<int> CLASSES_TO_KEEP = {0, 2, 76, 79}; // person
//Mouse callback for cv object:
// Special case for when point lies in a segment 
bool onSegment(Point p, Point q, Point r) {
    if (r.x <= max(p.x, q.x) && r.x >= min(p.x, q.x) &&
        r.y <= max(p.y, q.y) && r.y >= min(p.y, q.y))
        return true;
    return false;
}

// Function to find the orientation of the ordered triplet (p, q, r)
// 0 -> p, q and r are collinear
// 1 -> Clockwise
// 2 -> Counterclockwise
int orientation(Point p, Point q, Point r) {
    int val = (q.y - p.y) * (r.x - q.x) - (q.x - p.x) * (r.y - q.y);
    if (val == 0)
        return 0;
    //just a fancy way of writng an ''if'
    return (val > 0) ? 1 : 2;
}

// Function to check if two segments p1q1 and p2q2 intersect
bool segmentsIntersect(Point p1, Point q1, Point p2, Point q2) {
    int o1 = orientation(p1, q1, p2);
    int o2 = orientation(p1, q1, q2);
    int o3 = orientation(p2, q2, p1);
    int o4 = orientation(p2, q2, q1);

    // General case
    if (o1 != o2 && o3 != o4)
        return true;

    // Special cases
    if (o1 == 0 && onSegment(p1, q1, p2))
        return true;

    if (o2 == 0 && onSegment(p1, q1, q2))
        return true;

    if (o3 == 0 && onSegment(p2, q2, p1))
        return true;

    if (o4 == 0 && onSegment(p2, q2, q1))
        return true;

    return false;
}

// Function to check if the quadrilateral intersects itself
bool isSelfIntersecting(const vector<Point>& quadrilateral) {
    Point A = quadrilateral[0];
    Point B = quadrilateral[1];
    Point C = quadrilateral[2];
    Point D = quadrilateral[3];

    if (segmentsIntersect(A, B, C, D) || segmentsIntersect(B, C, D, A))
        return true;

    return false;
}

void select_roi(int event, int x, int y, int flags, void* userdata) {
    if (event == EVENT_LBUTTONDOWN && current_roi.size() < 4) {
        current_roi.push_back(Point(x, y));
        if (current_roi.size() == 4) {
            if (isSelfIntersecting(current_roi)) {
                //std::cerr << "The lines cross each other" << std::endl;
                current_roi.clear();
                //clear current ROI and do not store if shape is invalid, i.e., it is self intercepting. 
            }
            else {
                //std::cout << "Valid shape" << std::endl;
                rois.push_back(current_roi);
                current_roi.clear();
                roi_count++;
                //clear current ROI and store the shape. Increase count. 
            }
            
        }
    }
}

void interpolateDetections(const vector<Detection>& previous_detections, const vector<Detection>& current_detections, vector<InterpolatedDetection>& interpolated_detections, float alpha) {
    interpolated_detections.clear();

    for (const auto& prev_det : previous_detections) {
        auto it = find_if(current_detections.begin(), current_detections.end(), [&](const Detection& det) {
            return det.class_id == prev_det.class_id;
            });

        if (it != current_detections.end()) {
            InterpolatedDetection interpolated;
            interpolated.bbox.x = static_cast<int>(prev_det.bbox.x * (1.0f - alpha) + it->bbox.x * alpha);
            interpolated.bbox.y = static_cast<int>(prev_det.bbox.y * (1.0f - alpha) + it->bbox.y * alpha);
            interpolated.bbox.width = static_cast<int>(prev_det.bbox.width * (1.0f - alpha) + it->bbox.width * alpha);
            interpolated.bbox.height = static_cast<int>(prev_det.bbox.height * (1.0f - alpha) + it->bbox.height * alpha);
            interpolated.class_name = prev_det.class_name;
            interpolated.confidence = prev_det.confidence * (1.0f - alpha) + it->confidence * alpha;
            interpolated_detections.push_back(interpolated);
        }
        else {
            interpolated_detections.push_back({ prev_det.bbox, prev_det.class_name, prev_det.confidence });
        }
    }
}


//Function to check if points of polygon lie inside Bounding Box
bool checkIntersection(const Rect& bbox, const vector<Point>& polygon) {
    //use the coordinates to find the center of the quadrilateral. 
    Point p1 = polygon[0];
    Point p2 = polygon[1];
    Point p3 = polygon[2];
    Point p4 = polygon[3];

    double x_center = (p1.x + p2.x + p3.x + p4.x) / 4.0;
    double y_center = (p1.y + p2.y + p3.y + p4.y) / 4.0;
    Point center = { static_cast<int>(x_center), static_cast<int>(y_center) };

    if (bbox.contains(center)) {
        return true;
    }
    for (const auto& point : polygon) {
        if (bbox.contains(point)) {
            return true;
        }
    }
    return false;
}
//Updated: New function to find if anyy of the edges of the bounding box lie within the contour of
//the polygon using PointPolygonTest
//Function to check if points lie in contour
bool checkInsideContour(const Rect& bbox, const vector<Point>& polygon) {
    // Define the four corners of the bounding box
    vector<Point> bboxCorners = {
        Point(bbox.x, bbox.y),                   // Top-left corner
        Point(bbox.x + bbox.width, bbox.y),      // Top-right corner
        Point(bbox.x, bbox.y + bbox.height),     // Bottom-left corner
        Point(bbox.x + bbox.width, bbox.y + bbox.height) // Bottom-right corner
    };

    // Check each corner to see if it lies inside the polygon
    for (const Point& corner : bboxCorners) {
        double result = pointPolygonTest(polygon, corner, false);
        if (result >= 0) { // result >= 0 indicates the point is inside or on the edge (= 0) of the polygon
            return true;
        }
    }

    return false;
}

Mat drawInterpolatedLabels(const Mat& image, const vector<InterpolatedDetection>& detections) {
    Mat result = image.clone();
    //Check if no detections on current frame:
    if (detections.empty()) {
        for (const auto& roi : rois) {
            result = drawROIpolygons(result, roi, Scalar(0, 255, 0));
        }
        return result;
    }
    //initialize counters of detections in ROI
    int detection_counter[] = { 0, 0, 0 };
    for (const auto& detection : detections) {
        //initialize iteration variable for detection_counter
        int i = 0;
        //start only if there are more than 0 rois drawn 
        if(roi_count>0){
            for (const auto& roi : rois) {
                bool interferes = checkInsideContour(detection.bbox, roi) || checkIntersection(detection.bbox, roi);
                if (interferes) {
                    detection_counter[i]++;
                }
                i++;
            }
        }

        //Bounding box:
        Scalar color = Scalar(0, 255, 0);
        cv::rectangle(result, detection.bbox, color, 2);
        string label = detection.class_name + ": " + to_string(detection.confidence);

        int baseLine;
        Size labelSize = getTextSize(label, FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);

        cv::rectangle(
            result,
            Point(detection.bbox.x, detection.bbox.y - labelSize.height),
            Point(detection.bbox.x + labelSize.width, detection.bbox.y + baseLine),
            Scalar(255, 255, 255),
            FILLED);

        cv::putText(
            result,
            label,
            Point(detection.bbox.x, detection.bbox.y),
            FONT_HERSHEY_SIMPLEX,
            0.5,
            Scalar(0, 0, 0),
            1);
    }
    //check if ROI state changed, and log it:
    if (roi_count > 0) {
        int i = 0;
        bool current_state[] = { 0, 0, 0 };
        for (const auto& roi : rois) {
            if (detection_counter[i] > 0) {
                result = drawROIpolygons(result, roi, Scalar(255, 0, 0));
                current_state[i] = 1;
                //cout << "Detections found in "<<roi_names[i]<<": " << detection_counter[i] << endl;
            }
            else {
            result = drawROIpolygons(result, roi, Scalar(0, 255, 0));
            }
            //log event if ROI state changed:
            if (current_state[i] != prev_state[i]) {
                current_time = Timestamp::getCurrentTimeStamp();
                cout << "Changed detected in " << roi_names[i] << " @ " << current_time << endl;
                roiEvents[roi_names[i]].emplace_back(current_time, current_state[i]);
                //update list with previous values:
                prev_state[i] = current_state[i];
            }
            i++;
        }
    }
    return result;
}

int main() {
    try {
        //Find the model for inference
        wostringstream model_path;
        model_path << L"yolov10n.onnx";
        InferenceEngine engine(model_path);

        //setup the camera, wiindow name is "Webcam"
        namedWindow("Webcam", WINDOW_AUTOSIZE);
        setMouseCallback("Webcam", select_roi, nullptr);
        VideoCapture cap(1); //changed to use secondary camera. (default is 0; cap(0))
        if (!cap.isOpened()) {
            cerr << "Error opening video stream" << endl;
            return -1;
        }

        int frame_skip = 5;
        int frame_count = 0;
        vector<Detection> previous_detections;
        vector<InterpolatedDetection> interpolated_detections;
        float alpha = 0.01f;  // Smoothing factor
        bool current_state = false; //no interference at start
        Mat frame;
        while (true) {
            cap >> frame;
            if (frame.empty()) break;

            frame_count++;
            if (frame_count % frame_skip == 0) {
                auto input_tensor_values = engine.preprocessImage(frame);
                auto results = engine.runInference(input_tensor_values);
                auto detections = engine.filterDetections(results, 0.5, 640, 640, frame.cols, frame.rows);

                // Filter detections to keep only the specified classes
                detections.erase(remove_if(detections.begin(), detections.end(), [](const Detection& det) {
                    return CLASSES_TO_KEEP.find(det.class_id) == CLASSES_TO_KEEP.end();
                    }), detections.end());

                previous_detections = detections;
                interpolateDetections(previous_detections, detections, interpolated_detections, alpha);
                Mat output_frame = drawInterpolatedLabels(frame, interpolated_detections);
                if (!current_roi.empty()) {
                    polylines(output_frame, current_roi, true, Scalar(0, 0, 255), 2); //red
                }
                //specify the window for output frame:
                imshow("Webcam", output_frame);
            }
            else {
                interpolateDetections(previous_detections, previous_detections, interpolated_detections, alpha);
                Mat output_frame = drawInterpolatedLabels(frame, interpolated_detections);
                if (!current_roi.empty()) {
                    polylines(output_frame, current_roi, true, Scalar(0, 0, 255), 2); //red
                }
                imshow("Webcam", output_frame);
            }
            //check if the state (inside or outside) has changed. Print the time if changed. 

            if (waitKey(1) == 27) break;  // Exit on ESC key
        }

    }
    catch (const Ort::Exception& e) {
        cerr << "ONNX Runtime Exception: " << e.what() << endl;
    }
    catch (const cv::Exception& e) {
        cerr << "OpenCV Exception: " << e.what() << endl;
    }
    catch (const std::exception& e) {
        cerr << "Exception: " << e.what() << endl;
    }

    //write to CSV after execution
    writeEventsToCSV(roiEvents, "events.csv");
    return 0;
}
