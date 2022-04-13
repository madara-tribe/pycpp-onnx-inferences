#include <assert.h>
#include <algorithm>
#include <dirent.h>
#include <stdio.h>
#include <stdlib.h>
#include <atomic>
#include <sys/stat.h>
#include <unistd.h>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <iostream>
#include <iomanip>
#include <queue>
#include <string>
#include <vector>
#include <thread> 
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <dnndk/dnndk.h>
#include <mutex>  
std::mutex mtx_;

using namespace std;
using namespace std::chrono;
using namespace cv;


// constants for segmentation network
#define KERNEL_CONV       "unet"
#define CONV_INPUT_NODE   "conv2d_1_convolution"
#define CONV_OUTPUT_NODE  "conv2d_transpose_2_conv2d_transpose"
#define IMAGEDIR "./seg_test_images/"
#define OUTPUT_DIR "./output/"
#define DPU_MODE_NORMAL 0
#define DPU_MODE_PROF   1
#define DPU_MODE_DUMP   2
#define CLS 5
#define IMG_HEIGHT 400
#define IMG_WIDTH 640
#define ORIG_HEIGHT 1216
#define ORIG_WIDTH 1936
#define THREADS 3
#define BLOCK_SIZE 10
#define SLEEP 1

uint8_t colorB[] = {0, 0, 255, 255, 69};
uint8_t colorG[] = {0, 0, 255, 0, 47};
uint8_t colorR[] = {0, 255, 0, 0, 142};

int image_num;
std::vector<string> img_filenames;
int t_cnt = 0;

void barrier(int tid){
    {
        std::lock_guard<std::mutex> lock(mtx_);
        t_cnt++;
    }
    while(1){
        {
            std::lock_guard<std::mutex> lock(mtx_);
            if(t_cnt % THREADS == 0) break;
        }
        usleep(SLEEP);
    }
}


vector<string> ListImages(const char *path) {
  vector<string> images;
  images.clear();
  struct dirent *entry;

  struct stat s;
  lstat(path, &s);
  if (!S_ISDIR(s.st_mode)) {
    fprintf(stderr, "Error: %s is not a valid directory!\n", path);
    exit(1);
  }

  DIR *dir = opendir(path);
  if (dir == nullptr) {
    fprintf(stderr, "Error: Open %s path failed.\n", path);
    exit(1);
  }

  while ((entry = readdir(dir)) != nullptr) {
    if (entry->d_type == DT_REG || entry->d_type == DT_UNKNOWN) {
      string name = entry->d_name;
      string ext = name.substr(name.find_last_of(".") + 1);
      if ((ext == "JPEG") || (ext == "jpeg") || (ext == "JPG") ||
          (ext == "jpg") || (ext == "PNG") || (ext == "png")) {
          images.push_back(name);
      }
    }
  }

  closedir(dir);
  sort(images.begin(), images.end());
  return images;
}


inline double etime_sum(timespec ts02, timespec ts01){
    return (ts02.tv_sec+(double)ts02.tv_nsec/(double)1000000000)
            - (ts01.tv_sec+(double)ts01.tv_nsec/(double)1000000000);
}


Mat clahe_preprocess(const Mat &image, Ptr<CLAHE> clahe){
    Mat lab_image;
    cvtColor(image, lab_image, CV_BGR2Lab);
    std::vector<Mat> lab_planes(3);
    split(lab_image, lab_planes); 
    Mat dst;
    clahe->apply(lab_planes[0], dst);

    // Merge the the color planes back into an Lab image
    dst.copyTo(lab_planes[0]);
    merge(lab_planes, lab_image);
    Mat image_clahe;
    cvtColor(lab_image, image_clahe, CV_Lab2BGR);
    return image_clahe;
}

void PostProc(const float *data, int height, int width, int depth, const char *output_filename){
  assert(data);
  std::string save_name = output_filename;
  Mat segMat(height, width, CV_8UC3);
  for (int row = 0; row < height; row++) {
    for (int col = 0; col < width; col++) {
      int i = row * width * depth + col * depth;
      auto max_ind = max_element(data + i, data + i + depth);
      int posit = distance(data + i, max_ind);
      segMat.at<Vec3b>(row, col) = Vec3b(colorB[posit], colorG[posit], colorR[posit]);
    }
  }
  cvtColor(segMat, segMat, COLOR_BGR2RGB);
  // cv::resize(segMat, segMat, Size(ORIG_WIDTH, ORIG_HEIGHT), INTER_NEAREST); Do resize at python
  imwrite(OUTPUT_DIR + save_name.erase(8)+".png", segMat);
}


inline void set_input_image(DPUTask *task, int width, const Mat& image)
{
  //Mat cropped_img;
  DPUTensor* dpu_in = dpuGetInputTensor(task, CONV_INPUT_NODE);
  float scale       = dpuGetTensorScale(dpu_in);
  int8_t* data      = dpuGetTensorAddress(dpu_in);
  image.forEach<Vec3b>([&](Vec3b &p, const int *pos) -> void{
      int start = pos[0]*width*3+pos[1]*3;
      for(int k=0; k <3; k++){
        data[start+k] = (float(image.at<Vec3b>(pos[0],pos[1])[k])/255)* scale;
      }
  });
}


int main_thread(DPUKernel *kernelConv, int s_num, int e_num, int tid){
  assert(kernelConv);
  DPUTask *task = dpuCreateTask(kernelConv, DPU_MODE_NORMAL); 

  struct timespec ts01, ts02, ts03, ts04, ts05;
  double sum1 = 0, sum2 = 0, sum3 = 0, tsum = 0;

  string image_file_name[BLOCK_SIZE];
  Mat input_image[BLOCK_SIZE];
  const auto for_resize = Size(IMG_WIDTH, IMG_HEIGHT);

  DPUTensor *conv_out_tensor = dpuGetOutputTensor(task, CONV_OUTPUT_NODE);
  int outHeight = dpuGetTensorHeight(conv_out_tensor);
  int outWidth  = dpuGetTensorWidth(conv_out_tensor);
  int outChannel= dpuGetOutputTensorChannel(task, CONV_OUTPUT_NODE);
  int outSize = dpuGetOutputTensorSize(task, CONV_OUTPUT_NODE);
  float outScale = dpuGetOutputTensorScale(task, CONV_OUTPUT_NODE);
  
  Ptr<CLAHE> clahes = createCLAHE();
  clahes->setClipLimit(10);
  clahes->setTilesGridSize(Size(8, 8)); 

  // Main Loop
  int cnt=0;
  for(cnt=s_num; cnt<=e_num; cnt+=BLOCK_SIZE){
      for(int i=0; i<BLOCK_SIZE;i++){
        if(cnt+i>e_num) break;
        image_file_name[i] = img_filenames[cnt+i];
        input_image[i] = imread(IMAGEDIR+image_file_name[i]);
        if (input_image[i].empty()) {
            printf("cannot load %s\n", image_file_name[i].c_str());
            abort();
        }
      }
      
      barrier(tid);

      usleep(1000);
      //clock_gettime(CLOCK_REALTIME, &ts01);
      barrier(tid);

      for(int i=0; i<BLOCK_SIZE;i++){
        if(cnt+i>e_num) break;
        //cout << "filename : " << image_file_name[i] << endl;
        clock_gettime(CLOCK_REALTIME, &ts02);
        // resize
        Mat img;
        resize(input_image[i], img, for_resize, INTER_NEAREST);
        // pre-process with histgram avaraving
        Mat clahe_img = img;
        if((int)mean(img)[0] < 80) {
           clahe_img = clahe_preprocess(img, clahes);	
        }
        float *softmax = new float[outWidth*outHeight*outChannel];
      
        // Set image into Conv Task with mean value
        set_input_image(task, outWidth, clahe_img);
        {
          std::lock_guard<std::mutex> lock(mtx_);
          clock_gettime(CLOCK_REALTIME, &ts03);
          sum1 += etime_sum(ts03,ts02);
          dpuRunTask(task);
        }
        {
          std::lock_guard<std::mutex> lock(mtx_);
          //cout << "outScale : " << outScale << endl;
          int8_t *outAddr = (int8_t *)dpuGetOutputTensorAddress(task, CONV_OUTPUT_NODE);
          dpuRunSoftmax(outAddr, softmax, outChannel,outSize/outChannel, outScale);
          clock_gettime(CLOCK_REALTIME, &ts04);
          sum2 += etime_sum(ts04,ts03);
        }

        // Post process
        PostProc(softmax, outHeight, outWidth, outChannel, image_file_name[i].c_str());
        clock_gettime(CLOCK_REALTIME, &ts05);
        sum3 += etime_sum(ts05,ts04);
        delete[] softmax;
        tsum += etime_sum(ts05,ts02);
      }

      barrier(tid);
      cout << "new BLOCK loop " << endl;
      //clock_gettime(CLOCK_REALTIME, &ts06);
      //sum4 += etime_sum(ts06, ts01);
  }
  dpuDestroyTask(task);
  printf("sum1 : avarage prepro time: %8.3lf[ms]\n", (float)sum1/image_num*1000);
  printf("sum2 : avarage predict time: %8.3lf[ms]\n", (float)sum2/image_num*1000);
  printf("sum3 : avarage postpro time: %8.3lf[ms]\n", (float)sum3/image_num*1000);
  printf("tsum : avarage total time: %8.3lf[ms]\n", (float)tsum/image_num*1000);
  printf("FPS        : %8.3lf (%8.3lf [ms])\n", (float)image_num/tsum, (float)tsum/image_num*1000);   

  int tmp = image_num%(THREADS*BLOCK_SIZE);
  //printf("%d %d\n", tid, tmp);
  if(tid >= tmp){
      usleep(SLEEP);
      barrier(tid);
      usleep(SLEEP);
      barrier(tid);
      usleep(SLEEP);
      barrier(tid);
   }
   return 0;
}


int main(int argc, char **argv){
  DPUKernel *kernelConv;
  cout << "now running " << argv[0] << endl;
  img_filenames = ListImages(IMAGEDIR);
  if (img_filenames.size() == 0) {
    cout << "\nError: Not images exist in " << IMAGEDIR << endl;
  } else {
    image_num = img_filenames.size();
    cout << "total image : " << img_filenames.size() << endl;
  }
  
  int th_srt[THREADS];
  int th_end[THREADS];
  th_srt[0] = 0;
  th_end[0] = image_num / THREADS;
  if((image_num%THREADS)==0) {
      th_end[0]--;
  }
  for(int i=1;i<THREADS;i++){
      th_srt[i] = th_end[i-1]+1;
      th_end[i] = th_srt[i]+(image_num / THREADS);
      if(i>=(image_num%THREADS)){
          th_end[i]--;
      }
  }

  for(int i=0;i<THREADS;i++){
      printf("th_srt[%d] = %d, th_end[%d] = %d\n", i, th_srt[i], i, th_end[i]);
  }

  dpuOpen();
  kernelConv = dpuLoadKernel(KERNEL_CONV);
  // Parallel processing
  vector<thread> ths;
  for (int i = 0; i < THREADS; i++){
      ths.emplace_back(thread(main_thread, kernelConv, th_srt[i], th_end[i], i));
  }

  for (auto& th: ths){
      th.join();
  }
  dpuDestroyKernel(kernelConv);
  dpuClose();
  cout << "\nFinished ..." << endl;
  return 0;
}
