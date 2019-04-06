//
// Created by aayush on 5/4/19.
//
#include "opencv2/objdetect/objdetect.hpp"
#include <iostream>
#include <opencv2/imgproc.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>

using namespace std;
using namespace cv;

inline uint8_t *getPixel(Mat &z, int x, int y) {
    return const_cast<uint8_t *>(z.datastart + (x + (y * z.cols)) * z.channels());
}

void getSkin(Mat &frame, Mat &dest) {
    cvtColor(frame, frame, COLOR_BGR2GRAY);
    threshold(frame, dest, 110, 255, THRESH_BINARY);
}

void removePalm(Mat &img) {
    Mat n;
    Mat kernel = Mat::ones(7, 7, CV_8U);
    erode(img, n, kernel, Point(-1, -1), 10);
    dilate(n, n, kernel, Point(-1, -1), 14);
    Mat nn = img - n;
    nn.copyTo(img);
}

void getBG(Mat &bgdest, VideoCapture &cap, Rect2d region) {
    int frames = 0;
    Mat frame;
    cap >> frame;
    Mat bg = Mat::zeros(static_cast<int>(region.height), static_cast<int>(region.width), CV_32SC3);
    while (cap.isOpened() && frames <= 100) {
        cap >> frame;
        frame = frame(region);
        GaussianBlur(frame, frame, Size(5, 5), 10);
        frame.convertTo(frame, CV_32SC3);
        add(bg, frame, bg);
        frames++;
    }
    Mat bgfinal = bg / 100;
    bgfinal.convertTo(bgfinal, CV_8UC3);
    bgfinal.copyTo(bgdest);
}

void subtractBG(Mat &roi, Mat &bg, Mat &dest) {
    GaussianBlur(roi, roi, Size(5, 5), 10);
    absdiff(roi, bg, roi);
    cvtColor(roi, roi, COLOR_BGR2GRAY);
    threshold(roi, dest, 20, 255, THRESH_BINARY);
}

void denoiseBinary(Mat &img) {
    Mat kernel = Mat::ones(3, 3, CV_8U);
    erode(img, img, kernel, Point(-1, -1), 3);
    dilate(img, img, kernel, Point(-1, -1), 3);
    dilate(img, img, kernel, Point(-1, -1), 3);
    erode(img, img, kernel, Point(-1, -1), 3);

}

//void getSkin(Mat &img, Mat &dest) {
//    cvtColor(img, img, COLOR_BGR2GRAY);
//    adaptiveThreshold(img, dest, 255, THRESH_OTSU, THRESH_BINARY, 15, 1.0);
//}

void getCropped(Mat &img, Mat &dest) {
    int l = 0, r = img.cols - 1, t = img.rows - 1;
    for (int x = 0; x < img.cols; x++) {
        bool found = false;
        for (int y = 0; y < img.rows; y++) {
            if (*(getPixel(img, x, y)) > 0) {
                found = true;
                l = x;
                break;
            }
        }
        if (found) { break; }
    }
    for (int u = img.cols - 1; u > 0; u--) {
        bool found = false;
        for (int y = 0; y < img.rows; y++) {
            if (*(getPixel(img, u, y)) > 0) {
                found = true;
                r = u;
                break;
            }
        }
        if (found) { break; }
    }
    for (int y = 0; y < img.rows; y++) {
        bool found = false;
        for (int x = 0; x < img.cols; x++) {
            if (*(getPixel(img, x, y)) > 0) {
                found = true;
                t = y;
                break;
            }
        }
        if (found) { break; }
    }
    Mat z;
    z = img(Rect2d(Point(l, t), Point(r, img.rows - 1)));
    if (!(z.cols == 0 && z.rows == 0)) {
        resize(z, dest, Size(128, 128));
    }
}

void getResults(Mat &hand, int *dest) {
    int top_count = 0;
    bool top_in = false;
    for (int i = 0; i < 128; i++) {
        //20 pixels from top line
        if (!top_in && *(getPixel(hand, i, 20)) > 10) {
            top_in = true;
        } else if (top_in && *(getPixel(hand, i, 20)) == 0) {
            top_in = false;
            top_count++;
        }
    }
    dest[0] = top_count;
    int top_end = 0;
    if (top_in) { top_end++; }
    dest[1] = top_end;
    int left_count = 0;
    int left_pix = 0;
    bool left_in = false;
    for (int i = 128; i > 0; i--) {
        //10 pixels from left line
        //*(getPixel(hand, 10, i))=255;
        if (!left_in && *(getPixel(hand, 20, i)) > 10) {
            left_in = true;
        } else if (left_in && *(getPixel(hand, 20, i)) == 0) {
            left_in = false;
            left_count++;
        }
    }
    dest[2] = left_count;
    int left_end = 0;
    if (left_in) { left_end++; }
    dest[3] = left_end;
    int right_count = 0;
    bool right_in = false;
    for (int i = 128; i > 0; i--) {
        //10 pixels from right line
        //*(getPixel(hand, 10, i))=255;
        if (!right_in && *(getPixel(hand, 108, i)) > 10) {
            right_in = true;
        } else if (right_in && *(getPixel(hand, 108, i)) == 0) {
            right_in = false;
            right_count++;
        }
    }
    dest[4] = right_count;
    int right_end = 0;
    if (right_in) { right_end++; }
    dest[5] = right_end;
    double s = sum(hand)[0];
    dest[6] = static_cast<int>(s);
}

int parseResults(int *results) {
    cout << results[0] << "," << results[1] << "," << results[2] << "," << results[3] << "," << results[4] << ","
         << results[5] << "," << results[6] << "\n";
}

int getNumber(int *a) {
    if (a[0] == 1 && a[1] == 0 && a[2] == 2 && a[3] == 0 && a[4] == 2 && a[5] == 0 && a[6]/100000 == 22) { return 1; }
    if (a[0] == 2 && a[1] == 0 && a[2] == 2 && a[3] == 0 && a[4] == 2 && a[5] == 0 && a[6]/100000 == 21) { return 2; }
    if (a[0] == 2 && a[1] == 0 && a[2] == 2 && a[3] == 1 && a[4] == 2 && a[5] == 0 && a[6]/100000 < 20) { return 3; }
    if (a[0] == 1 && a[1] == 0 && a[2] == 2 && a[3] == 0 && a[4] == 2 && a[5] == 0 && a[6]/100000 == 19) { return 3; }
    if (a[0] == 3 && a[1] == 0 && a[2] == 2 && a[3] == 0 && a[4] == 2 && a[5] == 0 && a[6]/100000 == 22) { return 8; }
    if (a[0] == 3 && a[1] == 0 && a[2] == 2 && a[3] == 0 && a[4] == 2 && a[5] == 0 && a[6]/100000 > 21) { return 4; }
    if (a[0] == 3 && a[1] == 0 && a[2] == 2 && a[3] == 0 && a[4] == 3 && a[5] == 0 && a[6]/100000 > 21) { return 4; }
    if (a[0] == 3 && a[1] == 0 && a[2] == 2 && a[3] == 0 && a[4] == 2 && a[5] == 0 && a[6]/100000 < 21) { return 5; }
    if (a[0] == 3 && a[1] == 0 && a[2] == 4 && a[3] == 0 && a[4] == 2 && a[5] == 0) { return 6; }
    if (a[0] == 2 && a[1] == 0 && a[2] == 2 && a[3] == 0 && a[4] == 2 && a[5] == 0 && a[6]/100000 < 21) { return 7; }
    if (a[0] == 2 && a[1] == 0 && a[2] == 2 && a[3] == 0 && a[4] == 2 && a[5] == 0 && a[6]/100000 == 22) { return 8; }
    if (a[0] == 2 && a[1] == 0 && a[2] == 2 && a[3] == 0 && a[4] == 2 && a[5] == 0 && a[6]/100000 == 23) { return 9; }
    return -1;
}

int main() {
    VideoCapture cap("v.mp4");
    Mat frame;
    Mat d;
    Mat n;
    int results[7];
    while (cap.isOpened()) {
        cap >> frame;
        flip(frame, frame, 0);
        getSkin(frame, d);
        denoiseBinary(d);
        //removePalm(d);
        getCropped(d, n);
        getResults(n, results);
        parseResults(results);
        //cout<<getNumber(results)<<"\n";
        stringstream str1;
        str1<<getNumber(results);
        putText(frame, str1.str(), Point(frame.cols/2, frame.rows/2), FONT_HERSHEY_SIMPLEX, 4, Scalar(255, 255, 255));
        imshow("frame", frame);
        if (n.cols > 0) {
            imshow("dest", n);
        }
        waitKey(1);
    }
}