#include <fstream>
#include <iostream>
#include <vector>
#include <chrono>
#include <iomanip>
#include <string>
#include <cstring>
#include <algorithm>
#include <numeric>
#include <cuda_fp16.h>
typedef half value_t;
#define zero (__float2half(0.0f))

// 读取MNIST数据集
std::vector<std::vector<value_t>> read_mnist_images(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        std::cout << "Cannot open file!" << std::endl;
        return {};
    }

    int magic_number = 0, num_images = 0, num_rows = 0, num_cols = 0;
    file.read((char*)&magic_number, sizeof(magic_number));
    file.read((char*)&num_images, sizeof(num_images));
    file.read((char*)&num_rows, sizeof(num_rows));
    file.read((char*)&num_cols, sizeof(num_cols));

    // Reverse Integers (MNIST data is in big endian format)
    magic_number = ((magic_number & 0xff000000) >> 24) | ((magic_number & 0x00ff0000) >> 8) |
                   ((magic_number & 0x0000ff00) << 8) | ((magic_number & 0x000000ff) << 24);
    num_images = ((num_images & 0xff000000) >> 24) | ((num_images & 0x00ff0000) >> 8) |
                 ((num_images & 0x0000ff00) << 8) | ((num_images & 0x000000ff) << 24);
    num_rows = ((num_rows & 0xff000000) >> 24) | ((num_rows & 0x00ff0000) >> 8) |
                ((num_rows & 0x0000ff00) << 8) | ((num_rows & 0x000000ff) << 24);
    num_cols = ((num_cols & 0xff000000) >> 24) | ((num_cols & 0x00ff0000) >> 8) |
                ((num_cols & 0x0000ff00) << 8) | ((num_cols & 0x000000ff) << 24);

    int image_size = num_rows * num_cols;
    std::vector<std::vector<value_t>> images(num_images, std::vector<value_t>(image_size));

    for (int i = 0; i < num_images; ++i) {
        for (int j = 0; j < image_size; ++j) {
            unsigned char pixel = 0;
            file.read((char*)&pixel, sizeof(pixel));
            images[i][j] = static_cast<float>(pixel) / 255.0f;
        }
//		value_t mean = std::accumulate(images[i].begin(), images[i].end(), 0.0f) / images[i].size();
//		value_t std = std::sqrt(std::inner_product(images[i].begin(), images[i].end(), images[i].begin(), 0.0f) / images[i].size() - mean * mean);
//		std::transform(images[i].begin(), images[i].end(), images[i].begin(), [mean, std](value_t x) { 
//			return (x - mean) / std; 
//		});
    }

    return images;
}

// 读取MNIST label数据集
std::vector<int> read_mnist_labels(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        std::cout << "Cannot open file!" << std::endl;
        return {};
    }

    int magic_number = 0, num_items = 0;
    file.read((char*)&magic_number, sizeof(magic_number));
    file.read((char*)&num_items, sizeof(num_items));

    // Reverse Integers (MNIST data is in big endian format)
    magic_number = ((magic_number & 0xff000000) >> 24) | ((magic_number & 0x00ff0000) >> 8) |
                   ((magic_number & 0x0000ff00) << 8) | ((magic_number & 0x000000ff) << 24);
    num_items = ((num_items & 0xff000000) >> 24) | ((num_items & 0x00ff0000) >> 8) |
                ((num_items & 0x0000ff00) << 8) | ((num_items & 0x000000ff) << 24);

    std::vector<int> labels(num_items);
    for (int i = 0; i < num_items; ++i) {
        unsigned char label = 0;
        file.read((char*)&label, sizeof(label));
        labels[i] = static_cast<int>(label);
    }

    return labels;
}

// 读取模型参数
std::vector<value_t> read_param(const std::string& path) {
    std::ifstream file(path);
    std::vector<value_t> params;
    float param;
    while (file >> param) {
        params.push_back(param);
    }
    return params;
}

void move_param_to_gmem(value_t **dev_ptr, const std::vector<value_t>& host_vec) {
	cudaMalloc(dev_ptr, host_vec.size() * sizeof(value_t));
	cudaMemcpy(*dev_ptr, host_vec.data(), host_vec.size() * sizeof(value_t), cudaMemcpyHostToDevice);
}

__constant__ value_t const_conv1_weight[1 * 1 * 5 *5];
__constant__ value_t const_conv1_bias[1];

#define inputs(N,C,H,W) inputs_ptr[(((N) * inC + (C)) * inH + (H)) * inW + (W)]
#define outputs(N,C,H,W) outputs_ptr[(((N) * outC + (C)) * outH + (H)) * outW + (W)]
#define conv_weight(outc,inc,H,W) const_conv1_weight[(((outc) * inC + (inc)) * kH + (H)) * kW + (W)]
#define conv_bias(outc) const_conv1_bias[outc]
template<int inC, int inH, int inW, int outC, int outH, int outW, int kH, int kW, int outTile>
__global__ void conv1(value_t* inputs_ptr, value_t* outputs_ptr, int N){
	__shared__ value_t sm_inputs[inH][inW];
	int outh = threadIdx.x / outTile * 2;
	int outw = threadIdx.x % outTile * 2;
	for(int n = blockIdx.x; n < N; n+=gridDim.x){
		for(int outc = 0; outc < outC; outc++){
			value_t val[2][2];
			value_t reg_bias = conv_bias(outc);
			for(int outh2 = 0; outh2 < 2; outh2++){
				for(int outw2 = 0; outw2 < 2; outw2++){
					val[outh2][outw2] = reg_bias;
				}
			}
			for(int inc = 0; inc < inC; inc++){
				for(int pos = threadIdx.x; pos < inH * inW; pos+=blockDim.x){
					int inh = pos / inW; 
					int inw = pos % inW; 
					((value_t*)sm_inputs)[pos] = inputs(n, inc, inh, inw);
				}
				__syncthreads();
				for(int kh = 0; kh < kH; kh++){
					for(int kw = 0; kw < kW; kw++){
						value_t reg_weight = conv_weight(outc, inc, kh, kw);
						for(int outh2 = 0; outh2 < 2; outh2++){
							for(int outw2 = 0; outw2 < 2; outw2++){
								val[outh2][outw2] += sm_inputs[outh + outh2 + kh][outw + outw2 + kw] * reg_weight;
							}
						}
					}
				}
				__syncthreads();
			}
			value_t val2 = zero;
			for(int outh2 = 0; outh2 < 2; outh2++){
				for(int outw2 = 0; outw2 < 2; outw2++){
					val2 = __hmax(val2, val[outh2][outw2]);
				}
			}
			outputs(n, outc, outh / 2, outw / 2) = val2;
		}
	}
}
#undef conv_weight
#undef conv_bias
#undef inputs
#undef outputs

#define A(i,j) A[(i)*lda+(j)]
#define B(i,j) B[(i)*ldb+(j)]
#define C(i,j) C[(i)*ldc+(j)]
template<int MTile, int NTile, int KTile>
__global__ void matrixMul(const value_t* A, const value_t* B, value_t* C, const value_t* vec, int M, int N, int K) {
	__shared__ value_t sm_A[KTile][MTile], sm_B[KTile][NTile];
	const int lda = K, ldb = K, ldc = N;
	int bidn = blockIdx.x * NTile, bidm = blockIdx.y * MTile;
	int tid = threadIdx.y * blockDim.x + threadIdx.x;
	int tidn = threadIdx.x, tidm = threadIdx.y;
	value_t sum = 0;
	for (int k = 0; k < K; k += KTile) {
		for(int pos = tid; pos < KTile * MTile; pos += blockDim.y * blockDim.x){
			int ak = pos % KTile; 
			int am = pos / KTile; 
			sm_A[ak][am] = bidm + am < M ? A(bidm + am, k + ak) : zero;
		}
		for(int pos = tid; pos < KTile * NTile; pos += blockDim.y * blockDim.x){
			int bk = pos % KTile; 
			int bn = pos / KTile; 
			sm_B[bk][bn] = bidn + bn < N ? B(bidn + bn, k + bk) : zero;
		}
		__syncthreads();
		for (int j = 0; j < KTile; j++) {
			sum += sm_A[j][tidm] * sm_B[j][tidn];
		}
		__syncthreads();
	}
	if(bidm + tidm < M && bidn + tidn < N){
		C(bidm + tidm, bidn + tidn) = sum + vec[bidn + tidn];
	}
}
#undef A
#undef B
#undef C

int main(int argc, char* argv[]) {
	std::string dir = argv[1];  // 第一个参数是程序所在的目录，这个目录是存放前一步训练模型参数文件的目录，从这个目录下读取模型参数文件，相对于这个目录读取测试集图片和标签
	// cout << dir;
	
    // 读取测试集，对于想实现CUDA C/C++训练的同学，参考训练集文件名为train-images-idx3-ubyte和train-labels-idx1-ubyte
    auto images = read_mnist_images(dir + "/../../data/FashionMNIST/raw/t10k-images-idx3-ubyte");
    // 读取测试集标签
    auto labels = read_mnist_labels(dir + "/../../data/FashionMNIST/raw/t10k-labels-idx1-ubyte");
    // 读取模型参数
    auto conv_weight = read_param(dir + "/conv.weight.txt");
    auto conv_bias = read_param(dir + "/conv.bias.txt");
    auto fc_weight = read_param(dir + "/fc.weight.txt");
    auto fc_bias = read_param(dir + "/fc.bias.txt");

    // 打印每一个标签，仅用于调试！
	/*
    for (const auto& label : labels) {
        std::cout << label << " ";
    }
	std::cout<<std::endl;
	*/

	int num_images = images.size();
	value_t *host_images;
	cudaMallocHost(&host_images, num_images * 1 * 28 * 28 * sizeof(value_t));
	for(int i = 0; i < num_images; i++){
		std::memcpy(host_images + i * 1 * 28 * 28, images[i].data(), 1 * 28 * 28 * sizeof(value_t));
	}

	// 数据搬运
	value_t *dev_images, *dev_conv, *dev_labels;
	cudaMalloc(&dev_images, num_images * 1 * 28 * 28 * sizeof(value_t));
	cudaMalloc(&dev_conv, num_images * 1 * 13 * 13 * sizeof(value_t));
	cudaMalloc(&dev_labels, num_images * 10 * sizeof(value_t));
	cudaMemcpy(dev_images, host_images, num_images * 1 * 28 * 28 * sizeof(value_t), cudaMemcpyHostToDevice);
	cudaFreeHost(host_images);
	
	// 参数加载
    value_t *dev_fc_weight, *dev_fc_bias;
    move_param_to_gmem(&dev_fc_weight, fc_weight);
    move_param_to_gmem(&dev_fc_bias, fc_bias);
	cudaMemcpyToSymbol(const_conv1_weight, conv_weight.data(), conv_weight.size() * sizeof(value_t));
	cudaMemcpyToSymbol(const_conv1_bias, conv_bias.data(), conv_bias.size() * sizeof(value_t));
	value_t *host_labels;
	cudaMallocHost(&host_labels, num_images * 10 * sizeof(value_t));

    // 开始计时，使用chrono计时，不支持其它计时方式
    auto start = std::chrono::high_resolution_clock::now();
	conv1<1, 28, 28, 1, 13, 13, 3, 3, 13><<< 1024, 13 * 13 >>>(dev_images, dev_conv, num_images);
	matrixMul<16, 10, 13><<< dim3(1, (num_images + 15) / 16), dim3(10, 16) >>>(dev_conv, dev_fc_weight, dev_labels, dev_fc_bias, num_images, 10, 13 * 13); 
	cudaMemcpy(host_labels, dev_labels, num_images * 10 * sizeof(value_t), cudaMemcpyDeviceToHost);
	
	// 向主机端同步以等待所有异步调用的GPU kernel执行完毕，这句必须要有
	cudaDeviceSynchronize();

    // 结束计时
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end - start;

	int correct = 0;
	for(int i = 0; i < num_images; i++){
		int label = 0;
		for(int j = 1; j < 10; j++){
			if((float)host_labels[i * 10 + j] > (float)host_labels[i * 10 + label]){
				label = j;
			}
		}
		correct += (label == labels[i]);
	}

    // 输出结果，请严格保持此输出格式，并把0.0001替换成实际的准确率，请不要输出除了此结果之外的任何内容！！！
    std::cout << std::fixed << std::setprecision(4) << diff.count() << ":" << 1.0 * correct / num_images << std::endl;

	// 释放内存
	cudaFreeHost(host_labels);
	cudaFree(dev_images);
	cudaFree(dev_conv);
	cudaFree(dev_labels);

    cudaFree(dev_fc_weight);
    cudaFree(dev_fc_bias);

    return 0;
}
