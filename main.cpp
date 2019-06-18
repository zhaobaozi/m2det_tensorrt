#include "cudaUtility.h"
#include "mathFunctions.h"
#include "pluginImplement.h"
#include "tensorNet.h"
#include "loadImage.h"
#include "imageBuffer.h"
#include <chrono>
#include <thread>
#include<cmath>
#include <vector>
#include <algorithm>
#include <boost/algorithm/clamp.hpp>
#include<memory.h>
#include <map>
using namespace boost::algorithm;
const char* model  = "../../model/plugin_model/M2Det.prototxt";
const char* weight = "../../model/plugin_model/M2Det.caffemodel";

const char* INPUT_BLOB_NAME = "data";

const char* OUTPUT_BLOB_NAME = "out";
const char* OUTPUT1_BLOB_NAME = "reshape_pre";//reshape_pre
//const char* OUTPUT1_BLOB_NAME1 = "cls_perm_out";
static const uint32_t BATCH_SIZE = 1;
typedef struct Bbox
{
    int x1;
    int y1;
    int x2;
    int y2;
    float score;

}Bbox;
typedef struct Bbox_final
{
    int x1;
    int y1;
    int x2;
    int y2;
    float score;
    int cls;

}Bbox_final;
//image buffer size = 10
//dropFrame = false


//升序排列
bool cmpScore(Bbox lsh, Bbox rsh) {
	if (lsh.score < rsh.score)
		return true;
	else
		return false;
}
 
void nms(std::vector<Bbox> &boundingBox_, const float overlap_threshold){
 
    if(boundingBox_.empty()){
        return;
    }
    //对各个候选框根据score的大小进行升序排列
    sort(boundingBox_.begin(), boundingBox_.end(), cmpScore);
    float IOU = 0;
    float maxX = 0;
    float maxY = 0;
    float minX = 0;
    float minY = 0;
    std::vector<int> vPick;
    int nPick = 0;
    std::multimap<float, int> vScores;   //存放升序排列后的score和对应的序号
    const int num_boxes = boundingBox_.size();
	vPick.resize(num_boxes);
	for (int i = 0; i < num_boxes; ++i){
		vScores.insert(std::pair<float, int>(boundingBox_[i].score, i));
	}
    while(vScores.size() > 0){
        int last = vScores.rbegin()->second;  //反向迭代器，获得vScores序列的最后那个序列号
        vPick[nPick] = last;
        nPick += 1;
        for (std::multimap<float, int>::iterator it = vScores.begin(); it != vScores.end();){
            int it_idx = it->second;
            maxX = std::max(boundingBox_.at(it_idx).x1, boundingBox_.at(last).x1);
            maxY = std::max(boundingBox_.at(it_idx).y1, boundingBox_.at(last).y1);
            minX = std::min(boundingBox_.at(it_idx).x2, boundingBox_.at(last).x2);
            minY = std::min(boundingBox_.at(it_idx).y2, boundingBox_.at(last).y2);
            //转换成了两个边界框相交区域的边长
            maxX = ((minX-maxX+1)>0)? (minX-maxX+1) : 0;
            maxY = ((minY-maxY+1)>0)? (minY-maxY+1) : 0;
            //求交并比IOU
            
            IOU = (maxX * maxY)/((boundingBox_.at(it_idx).x2-boundingBox_.at(it_idx).x1)*(boundingBox_.at(it_idx).y2-boundingBox_.at(it_idx).y1) + (boundingBox_.at(last).x2-boundingBox_.at(last).x1)*(boundingBox_.at(last).y2-boundingBox_.at(last).y1) - IOU);
            if(IOU > overlap_threshold){
                it = vScores.erase(it);    //删除交并比大于阈值的候选框,erase返回删除元素的下一个元素
            }else{
                it++;
            }
        }
    }
    
    vPick.resize(nPick);
    std::vector<Bbox> tmp_;
    tmp_.resize(nPick);
    for(int i = 0; i < nPick; i++){
        tmp_[i] = boundingBox_[vPick[i]];
    }
    boundingBox_ = tmp_;
}

ConsumerProducerQueue<cv::Mat> *imageBuffer = new ConsumerProducerQueue<cv::Mat>(10,false);

class Timer {
public:
    void tic() {
        start_ticking_ = true;
        start_ = std::chrono::high_resolution_clock::now();
    }
    void toc() {
        if (!start_ticking_)return;
        end_ = std::chrono::high_resolution_clock::now();
        start_ticking_ = false;
        t = std::chrono::duration<double, std::milli>(end_ - start_).count();
        //std::cout << "Time: " << t << " ms" << std::endl;
    }
    double t;
private:
    bool start_ticking_ = false;
    std::chrono::time_point<std::chrono::high_resolution_clock> start_;
    std::chrono::time_point<std::chrono::high_resolution_clock> end_;
};


/* *
 * @TODO: unifiedMemory is used here under -> ( cudaMallocManaged )
 * */
float* allocateMemory(DimsCHW dims, char* info)
{
    float* ptr;
    size_t size;
    std::cout << "Allocate memory: " << info << std::endl;
    size = BATCH_SIZE * dims.c() * dims.h() * dims.w();
    assert(!cudaMallocManaged( &ptr, size*sizeof(float)));
    return ptr;
}


void loadImg( cv::Mat &input, int re_width, int re_height, float *data_unifrom,const float3 mean,const float scale )
{
    int i;
    int j;
    int line_offset;
    int offset_g;
    int offset_r;
    cv::Mat dst;

    unsigned char *line = NULL;
    float *unifrom_data = data_unifrom;

    cv::resize( input, dst, cv::Size( re_width, re_height ), (0.0), (0.0), cv::INTER_LINEAR );
    offset_g = re_width * re_height;
    offset_r = re_width * re_height * 2;
    for( i = 0; i < re_height; ++i )
    {
        line = dst.ptr< unsigned char >( i );
        line_offset = i * re_width;
        for( j = 0; j < re_width; ++j )
        {
            // b
            unifrom_data[ line_offset + j  ] = (( float )(line[ j * 3 ] - mean.x) * scale);
            // g
            unifrom_data[ offset_g + line_offset + j ] = (( float )(line[ j * 3 + 1 ] - mean.y) * scale);
            // r
            unifrom_data[ offset_r + line_offset + j ] = (( float )(line[ j * 3 + 2 ] - mean.z) * scale);
        }
    }
}

//thread read video
void readPicture()
{
    cv::VideoCapture cap("../../testVideo/test.avi");
    cv::Mat image;
    while(cap.isOpened())
    {
        cap >> image;
        imageBuffer->add(image);
    }
}

int main(int argc, char *argv[])
{
    std::vector<std::string> output_vector = {OUTPUT1_BLOB_NAME,OUTPUT_BLOB_NAME};
    TensorNet tensorNet;
    tensorNet.LoadNetwork(model,weight,INPUT_BLOB_NAME, output_vector,BATCH_SIZE);
    DimsCHW dimsData = tensorNet.getTensorDims(INPUT_BLOB_NAME);
    DimsCHW dimsOut  = tensorNet.getTensorDims(OUTPUT_BLOB_NAME);
    DimsCHW dimspre  = tensorNet.getTensorDims(OUTPUT1_BLOB_NAME);
    float* data    = allocateMemory( dimsData , (char*)"input blob");
    std::cout << "allocate data" << std::endl;
    float* output_pre    = allocateMemory( dimspre , (char*)"output_pre blob");
    std::cout << "allocate output_pre" << std::endl;
    float* output  = allocateMemory( dimsOut  , (char*)"output_cls blob");
    std::cout << "allocate output" << std::endl;
    int height = 512;
    int width  = 512;

    cv::Mat frame,srcImg;

    void* imgCPU;
    void* imgCUDA;
    Timer timer;
    //std::string imgFile = "/home/ubuntu/zhq/MobileNet-SSD-TensorRT/002840.jpg";
    //frame = cv::imread(imgFile);
    //std::cout<<frame<<std::endl;
    //std::thread readTread(readPicture);
    //readTread.detach();
   
    //imageBuffer->consume(frame);
    std::string imgFile = "/home/ubuntu/zhq/M2Det/data/VOCdevkit/VOC0712/JPEGImages/003335.jpg";
    frame = cv::imread(imgFile);
    int height_o,weight_o;
    height_o=frame.size().height;
    weight_o=frame.size().width;
    
    //std::cout<<"frame:"<<height_o<<weight_o<<std::endl;
    srcImg = frame.clone();
    cv::resize(frame, frame, cv::Size(512,512));
    const size_t size = width * height * sizeof(float3);

    if( CUDA_FAILED( cudaMalloc( &imgCUDA, size)) )
    {
        std::cout <<"Cuda Memory allocation error occured."<<std::endl;
        return false;
    }

    void* imgData = malloc(size);
    memset(imgData,0,size);

    loadImg(frame,height,width,(float*)imgData,make_float3(104, 117, 123),1);
    cudaMemcpyAsync(imgCUDA,imgData,size,cudaMemcpyHostToDevice);

    void* buffers[] = { imgCUDA, output_pre,output };

    timer.tic();
    double Time = (double)cvGetTickCount();
    tensorNet.imageInference( buffers, 3, BATCH_SIZE);

    
//prior
int img_width=512,img_height=512;
int steps[6]={8, 16, 32, 64, 128, 256};
float feature_maps[6]={64.0, 32.0, 16.0, 8.0, 4.0, 2.0};
float min_sizes[6]={30.72, 76.8, 168.96, 261.12, 353.28, 445.44};
float max_sizes[6]={76.8, 168.96, 261.12, 353.28, 445.44, 537.6};
int aspect_ratios[2]={2,3};
float variances[2]={0.1,0.2};
float f_k;
float s_k;
float s_k_prime;
int idx=0;
int size_1=131040;
float *mean=new float[size_1];//remeber delet
float *boxes=new float[size_1];//remeber delet
float *boxes_final=new float[size_1];//remeber delet
const float overlap_threshold=0.45;
for (int k = 0; k < 6; ++k) {
  for (int h = 0; h < feature_maps[k]; ++h) {
    for (int w = 0; w < feature_maps[k]; ++w) {
      f_k=img_width/steps[k];
      float center_x = (w + 0.5)/f_k;
      float center_y = (h + 0.5)/f_k;
      s_k=min_sizes[k]/img_width;
      mean[idx++]=center_x;
      mean[idx++]=center_y;
      mean[idx++]=s_k;
      mean[idx++]=s_k;
      s_k_prime=sqrt(s_k*(max_sizes[k]/img_width));
      mean[idx++]=center_x;
      mean[idx++]=center_y;
      mean[idx++]=s_k_prime;
      mean[idx++]=s_k_prime;
      for(int z = 0; z < 2; ++z){
      mean[idx++]=center_x;
      mean[idx++]=center_y;
      mean[idx++]=s_k*sqrt(aspect_ratios[z]);
      mean[idx++]=s_k/sqrt(aspect_ratios[z]);
      mean[idx++]=center_x;
      mean[idx++]=center_y;
      mean[idx++]=s_k/sqrt(aspect_ratios[z]);
      mean[idx++]=s_k*sqrt(aspect_ratios[z]);
      }
     }
    }
}

for(int j=0;j<32760;j++)
{
  for(int i=0;i<4;i++)
  {
      mean[j*4+i]=clamp(mean[j*4+i],0,1);
      if(i == 0 || i == 1)
        {
        boxes[j*4+i]=mean[j*4+i]+output_pre[j*4+i]*variances[0]*mean[j*4+i+2];

        }
      else if(i == 2 || i == 3)
        {
        //std::cout<<"output_pre:"<<output_pre[j*4+i]<<std::endl;
        boxes[j*4+i]=mean[j*4+i]*exp(output_pre[j*4+i]*variances[1]);
        }     
        
  }
}
std::cout<<"-----------------------"<<std::endl;
for(int i=0;i<10;i++)
{
std::cout<<"haha:"<<output_pre[i]<<std::endl;
}
for(int j=0;j<32760;j++)
{
  for(int i=0;i<4;i++)
  {
      if(i == 0 || i == 1)
        {
        boxes[j*4+i]-=boxes[j*4+i+2]/2;
       if(i==0)
          {
        boxes_final[j*4+i]= boxes[j*4+i]* weight_o;
          }
        else if (i==1)
          {
        boxes_final[j*4+i]= boxes[j*4+i]* height_o;
          }
        }
      else if(i == 2 || i == 3)
        {
        boxes[j*4+i]+=boxes[j*4+i-2];
        if(i==2)
          {
        boxes_final[j*4+i]= boxes[j*4+i]* weight_o;
          }
        else if (i==3)
          {
        boxes_final[j*4+i]= boxes[j*4+i]* height_o;
          }
        }     
        
  }
}

std::vector<Bbox> result_final;
Bbox BB1;
Bbox_final BB2;

std::vector<Bbox_final> BB_2;
for(int j=1;j<5;j++)
{
    std::vector<Bbox> BB_1;
    std::cout<<"--------------------------j------------------------------"<<j<<std::endl;
    for(int i=0;i<32760;i++)
    {
        if(output[5*i+j]>0.1)
          {
            BB1.x1=boxes_final[4*i+0];
            BB1.y1=boxes_final[4*i+1];
            BB1.x2=boxes_final[4*i+2];
            BB1.y2=boxes_final[4*i+3];
            BB1.score=output[5*i+j];
            BB_1.push_back(BB1);  
            nms(BB_1,overlap_threshold);
          }
    }

    for(int i=0;i<BB_1.size();i++)
    {
        BB2.x1=BB_1[i].x1;
std::cout<<"BB2.x1:"<<BB2.x1<<std::endl;
        BB2.y1=BB_1[i].y1;
std::cout<<"BB2.y1:"<<BB2.y1<<std::endl;
        BB2.x2=BB_1[i].x2;
std::cout<<"BB2.x2:"<<BB2.y1<<std::endl;
        BB2.y2=BB_1[i].y2;
std::cout<<"BB2.y2:"<<BB2.y2<<std::endl;
        BB2.score=BB_1[i].score;
std::cout<<"BB2.score:"<<BB2.score<<std::endl;
        BB2.cls=j;
std::cout<<"BB2.j:"<<j<<std::endl;
        BB_2.push_back(BB2);
        cv::rectangle(srcImg,cv::Rect2f(cv::Point(BB2.x1,BB2.y1),cv::Point(BB2.x2,BB2.y2)),cv::Scalar(255,0,255),1);

    }

}
    

    timer.toc();
    double msTime = timer.t;
    std::cout<<"---------------tensorrt_time:"<<msTime<<std::endl;
    cv::imshow("mobileNet",srcImg);
    cv::waitKey(4000);
    free(imgData);
    cudaFree(imgCUDA);
    cudaFreeHost(imgCPU);
    cudaFree(output);
    tensorNet.destroy();
    return 0;
}





