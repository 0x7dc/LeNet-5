// TestLeNet.cpp : 定义控制台应用程序的入口点。
//

#include "stdafx.h"
#include <stdlib.h>
#include <memory.h>
#include <math.h>
#include <time.h>
#include "mt19937ar.h"

#ifndef MIN
#define MIN(A,B)	(((A) <= (B)) ? (A) : (B))
#endif

#ifndef MAX
#define MAX(A,B)	(((A) >= (B)) ? (A) : (B))
#endif

#define uchar unsigned char

// 基本参数----------------------------------------------------------------------------------------------/
typedef struct _Sample
{
	double *data;
	double *label;

	int sample_w;
	int sample_h;
	int sample_count;
} Sample;

typedef struct _Kernel
{
	double *W;
	double *dW;
} Kernel;

typedef struct _Map
{
	double *data;
	double *error;
	double  b;
	double  db;
} Map;

typedef struct _Layer
{
	int map_w;
	int map_h;
	int map_count;
	Map *map;

	int kernel_w;
	int kernel_h;
	int kernel_count;
	Kernel *kernel;

	double *map_common;
} Layer;

const int batch_size = 10;
const int classes_count = 10;
const int width  = 32;
const int height = 32;
const int train_sample_count = 60000;
const int test_sample_count  = 10000;

Layer input_layer, output_layer;
Layer c1_conv_layer, c3_conv_layer, c5_conv_layer;
Layer s2_pooling_layer, s4_pooling_layer;

//*-------------------------------------------------------------------------------------------------------/

// 初始化------------------------------------------------------------------------------------------------/
void init_kernel(double *kernel, int size, double weight_base)
{
	for (int i = 0; i < size; i++)
	{
		kernel[i] = (genrand_real1() - 0.5) * 2 * weight_base;
	}
}

void init_layer(Layer *layer, int prevlayer_map_count, int map_count, int kernel_w, int kernel_h, int map_w, int map_h, bool is_pooling)
{
	int mem_size = 0;

	const double scale = 6.0;
	int fan_in = 0;
	int fan_out = 0;
	if (is_pooling)
	{
		fan_in  = 4;
		fan_out = 1;
	}
	else
	{
		fan_in = prevlayer_map_count * kernel_w * kernel_h;
		fan_out = map_count * kernel_w * kernel_h;
	}
	int denominator = fan_in + fan_out;
	double weight_base = (denominator != 0) ? sqrt(scale / (double)denominator) : 0.5;

	layer->kernel_count = prevlayer_map_count * map_count;
	layer->kernel_w = kernel_w;
	layer->kernel_h = kernel_h;
	layer->kernel = (Kernel *)malloc(layer->kernel_count * sizeof(Kernel));
	mem_size = layer->kernel_w * layer->kernel_h * sizeof(double);
	for (int i = 0; i < prevlayer_map_count; i++)
	{
		for (int j = 0; j < map_count; j++)
		{
			layer->kernel[i*map_count + j].W = (double *)malloc(mem_size);
			init_kernel(layer->kernel[i*map_count + j].W, layer->kernel_w*layer->kernel_h, weight_base);
			layer->kernel[i*map_count + j].dW = (double *)malloc(mem_size);
			memset(layer->kernel[i*map_count + j].dW, 0, mem_size);
		}
	}

	layer->map_count = map_count;
	layer->map_w = map_w;
	layer->map_h = map_h;
	layer->map = (Map *)malloc(layer->map_count * sizeof(Map));
	mem_size = layer->map_w * layer->map_h * sizeof(double);
	for (int i = 0; i < layer->map_count; i++)
	{
		layer->map[i].b = 0.0;
		layer->map[i].db = 0.0;
		layer->map[i].data = (double *)malloc(mem_size);
		layer->map[i].error = (double *)malloc(mem_size);
		memset(layer->map[i].data, 0, mem_size);
		memset(layer->map[i].error, 0, mem_size);
	}
	layer->map_common = (double *)malloc(mem_size);
	memset(layer->map_common, 0, mem_size);	
}

void release_layer(Layer *layer)
{
	for (int i = 0; i < layer->kernel_count; i++)
	{
		free(layer->kernel[i].W);
		free(layer->kernel[i].dW);
		layer->kernel[i].W = NULL;
		layer->kernel[i].dW = NULL;
	}
	free(layer->kernel);
	layer->kernel = NULL;

	for (int i = 0; i < layer->map_count; i++)
	{
		free(layer->map[i].data);
		free(layer->map[i].error);
		layer->map[i].data = NULL;
		layer->map[i].error = NULL;
	}
	free(layer->map_common);
	layer->map_common = NULL;
	free(layer->map);
	layer->map = NULL;
}
//*-------------------------------------------------------------------------------------------------------/

// 读取数据----------------------------------------------------------------------------------------------/
// 高低位转换
int SwapEndien_32(int i)
{
	return ((i & 0x000000FF) << 24) | ((i & 0x0000FF00) << 8) | ((i & 0x00FF0000) >> 8) | ((i & 0xFF000000) >> 24);
}

int ReverseInt(int i)
{
	unsigned char ch1, ch2, ch3, ch4;
	ch1 = i & 255;
	ch2 = (i >> 8)  & 255;
	ch3 = (i >> 16) & 255;
	ch4 = (i >> 24) & 255;

	return ((int)ch1<<24) + ((int)ch2<<16) + ((int)ch3<<8) + ch4;
}

// 读训练/测试集数据
void read_mnist_data(Sample *sample, const char *file_name)
{
	FILE *fp = NULL;
	fopen_s(&fp, file_name, "rb");

	int magic_number = 0;
	int sample_count = 0;
	int n_rows = 0, n_cols = 0, padding = 2;

	fread((char*)&magic_number, sizeof(magic_number), 1, fp);
	magic_number = SwapEndien_32(magic_number);
	fread((char*)&sample_count, sizeof(sample_count), 1, fp);
	sample_count = SwapEndien_32(sample_count);
	fread((char*)&n_rows, sizeof(n_rows), 1, fp);
	n_rows = SwapEndien_32(n_rows);
	fread((char*)&n_cols, sizeof(n_cols), 1, fp);
	n_cols = SwapEndien_32(n_cols);

	double scale_max =  1.0;
	double scale_min = -1.0;
	unsigned char temp = 0;
	int size = width*height;
	int mem_size = size*sizeof(double);
	for (int k = 0; k < sample_count; k++)
	{
		sample[k].data = (double *)malloc(mem_size);

		for (int i = 0; i < size; i++)
		{
			sample[k].data[i] = scale_min;
		}

		for (int i = 0; i < n_rows; i++)
		{
			for (int j = 0; j < n_cols; j++)
			{
				fread((char*)&temp, sizeof(temp), 1, fp);
				sample[k].data[(i + padding)*width + j + padding] = ((double)temp / 255.0) * (scale_max - scale_min) + scale_min;
			}
		}
	}

	fclose(fp);
	fp = NULL;
}

// 读训练/测试集标签
void read_mnist_label(Sample *sample, const char *file_name)
{
	FILE *fp = NULL;
	fopen_s(&fp, file_name, "rb");

	int magic_number = 0;
	int sample_count = 0;

	fread((char*)&magic_number, sizeof(magic_number), 1, fp);
	magic_number = SwapEndien_32(magic_number);

	fread((char*)&sample_count, sizeof(sample_count), 1, fp);
	sample_count = SwapEndien_32(sample_count);

	uchar label = 0;
	int mem_size = classes_count*sizeof(double);
	for (int k = 0; k < sample_count; k++)
	{
		sample[k].label = (double *)malloc(mem_size);
		for (int i = 0; i < classes_count; i++)
		{
			sample[k].label[i] = -0.8;
		}

		fread((char*)&label, sizeof(label), 1, fp);
		sample[k].label[label] = 0.8;
	}

	fclose(fp);
	fp = NULL;
}
//*-------------------------------------------------------------------------------------------------------/

// 损失函数----------------------------------------------------------------------------------------------/
struct loss_func 
{
	inline static double mse(double y, double t)
	{
		return (y - t) * (y - t) / 2;
	}

	inline static double dmse(double y, double t)
	{
		return y - t;
	}
};
//*-------------------------------------------------------------------------------------------------------/

// 激活函数----------------------------------------------------------------------------------------------/
struct activation_func
{
	/* scale: -0.8 ~ 0.8 和label初始值对应 */
	inline static double tan_h(double val)
	{
		double ep = exp(val);
		double em = exp(-val);

		return (ep - em) / (ep + em);
	}

	inline static double dtan_h(double val)
	{
		return 1.0 - val*val;
	}

	/* scale: 0.1 ~ 0.9 和label初始值对应 */
	inline static double relu(double val)
	{
		return val > 0.0 ? val : 0.0;
	}

	inline static double drelu(double val)
	{
		return val > 0.0 ? 1.0 : 0.0;
	}

	/* scale: 0.1 ~ 0.9 和label初始值对应 */
	inline double sigmoid(double val) 
	{ 
		return 1.0 / (1.0 + exp(-val)); 
	}

	double dsigmoid(double val)
	{ 
		return val * (1.0 - val); 
	}
};
//*-------------------------------------------------------------------------------------------------------/

// 克罗内克积--------------------------------------------------------------------------------------------/
void kronecker(double *in_data, int in_map_w, int in_map_h, double *out_data, int out_map_w)
{
	for (int i = 0; i < in_map_h; i++)
	{
		for (int j = 0; j < in_map_w; j++)
		{
			for (int n = 2*i; n < 2*(i + 1); n++)
			{
				for (int m = 2*j; m < 2*(j + 1); m++)
				{
					out_data[n*out_map_w + m] = in_data[i*in_map_w + j];
				}
			}
		}
	}
}
//*-------------------------------------------------------------------------------------------------------/

// 卷积--------------------------------------------------------------------------------------------------/
void convn_valid(double *in_data, int in_w, int in_h, double *kernel, int kernel_w, int kernel_h, double *out_data, int out_w, int out_h)
{
	double sum = 0.0;
	for (int i = 0; i < out_h; i++)
	{
		for (int j = 0; j < out_w; j++)
		{
			sum = 0.0;
			for (int n = 0; n < kernel_h; n++)
			{
				for (int m = 0; m < kernel_w; m++)
				{
					sum += in_data[(i + n)*in_w + j + m] * kernel[n*kernel_w + m];
				}
			}
			out_data[i*out_w + j] += sum;
		}
	}
}
//*-------------------------------------------------------------------------------------------------------/

// 正向传播----------------------------------------------------------------------------------------------/
#define O true
#define X false
bool connection_table[6*16] = 
{
	O, X, X, X, O, O, O, X, X, O, O, O, O, X, O, O,
	O, O, X, X, X, O, O, O, X, X, O, O, O, O, X, O,
	O, O, O, X, X, X, O, O, O, X, X, O, X, O, O, O,
	X, O, O, O, X, X, O, O, O, O, X, X, O, X, O, O,
	X, X, O, O, O, X, X, O, O, O, O, X, O, O, X, O,
	X, X, X, O, O, O, X, X, O, O, O, O, X, O, O, O
};
#undef O
#undef X

void conv_fprop(Layer *prev_layer, Layer *layer, bool *pconnection)
{
	int index = 0;
	int size = layer->map_w * layer->map_h;
	for (int i = 0; i < layer->map_count; i++)
	{
		memset(layer->map_common, 0, size*sizeof(double));
		for (int j = 0; j < prev_layer->map_count; j++)
		{
			index = j*layer->map_count + i;
			if (pconnection != NULL && !pconnection[index])
			{
				continue;
			}
		
			convn_valid(
				prev_layer->map[j].data, prev_layer->map_w, prev_layer->map_h, 
				layer->kernel[index].W, layer->kernel_w, layer->kernel_h, 
				layer->map_common, layer->map_w, layer->map_h);
		}

		for (int k = 0; k < size; k++)
		{
			layer->map[i].data[k] = activation_func::tan_h(layer->map_common[k] + layer->map[i].b);
		}
	}
}

void avg_pooling_fprop(Layer *prev_layer, Layer *layer)
{
	int map_w = layer->map_w;
	int map_h = layer->map_h;
	int upmap_w = prev_layer->map_w;
	const double scale_factor = 0.25;

	for (int k = 0; k < layer->map_count; k++)
	{
		for (int i = 0; i < map_h; i++)
		{
			for (int j = 0; j < map_w; j++)
			{
				double sum = 0.0;
				for (int n = 2*i; n < 2*(i + 1); n++)
				{
					for (int m = 2*j; m < 2*(j + 1); m++)
					{
						sum += prev_layer->map[k].data[n*upmap_w + m] * layer->kernel[k].W[0];
					}
				}

				sum *= scale_factor;
				sum += layer->map[k].b;
				layer->map[k].data[i*map_w + j] = activation_func::tan_h(sum);
			}
		}
	}
}

void max_pooling_fprop(Layer *prev_layer, Layer *layer)
{
	int map_w = layer->map_w;
	int map_h = layer->map_h;
	int upmap_w = prev_layer->map_w;

	for (int k = 0; k < layer->map_count; k++)
	{
		for (int i = 0; i < map_h; i++)
		{
			for (int j = 0; j < map_w; j++)
			{
				double max_value = prev_layer->map[k].data[2*i*upmap_w + 2*j];
				for (int n = 2*i; n < 2*(i + 1); n++)
				{
					for (int m = 2*j; m < 2*(j + 1); m++)
					{
						max_value = MAX(max_value, prev_layer->map[k].data[n*upmap_w + m]);
					}
				}

				layer->map[k].data[i*map_w + j] = activation_func::tan_h(max_value);
			}
		}
	}
}

void fully_connected_fprop(Layer *prev_layer, Layer *layer)
{
	for (int i = 0; i < layer->map_count; i++) 
	{
		double sum = 0.0;
		for (int j = 0; j < prev_layer->map_count; j++)
		{
			sum += prev_layer->map[j].data[0] * layer->kernel[j*layer->map_count + i].W[0];
		}

		sum += layer->map[i].b;
		layer->map[i].data[0] = activation_func::tan_h(sum);
	}
}

void forward_propagation()
{
	// In-->C1
	conv_fprop(&input_layer, &c1_conv_layer, NULL);

	// C1-->S2
	max_pooling_fprop(&c1_conv_layer, &s2_pooling_layer);/*avg_pooling_fprop*/

	// S2-->C3
	conv_fprop(&s2_pooling_layer, &c3_conv_layer, connection_table);

	// C3-->S4
	max_pooling_fprop(&c3_conv_layer, &s4_pooling_layer);/*avg_pooling_fprop*/

	// S4-->C5
	conv_fprop(&s4_pooling_layer, &c5_conv_layer, NULL);

	// C5-->Out
	fully_connected_fprop(&c5_conv_layer, &output_layer);
}
//*-------------------------------------------------------------------------------------------------------/

// 反向传播----------------------------------------------------------------------------------------------/
void conv_bprop(Layer *layer, Layer *prev_layer, bool *pconnection)
{
	int index = 0;
	int size = prev_layer->map_w * prev_layer->map_h;

	// delta
	for (int i = 0; i < prev_layer->map_count; i++)
	{
		memset(prev_layer->map_common, 0, size*sizeof(double));
		for (int j = 0; j < layer->map_count; j++)
		{
			index = i*layer->map_count + j;
			if (pconnection != NULL && !pconnection[index])
			{
				continue;
			}

			for (int n = 0; n < layer->map_h; n++)
			{
				for (int m = 0; m < layer->map_w; m++)
				{
					double error = layer->map[j].error[n*layer->map_w + m];
					for (int ky = 0; ky < layer->kernel_h; ky++)
					{
						for (int kx = 0; kx < layer->kernel_w; kx++)
						{
							prev_layer->map_common[(n + ky)*prev_layer->map_w + m + kx] += error * layer->kernel[index].W[ky*layer->kernel_w + kx];
						}
					}
				}
			}
		}

		for (int k = 0; k < size; k++)
		{
			prev_layer->map[i].error[k] = prev_layer->map_common[k] * activation_func::dtan_h(prev_layer->map[i].data[k]);
		}
	}

	// dW
	for (int i = 0; i < prev_layer->map_count; i++)
	{
		for (int j = 0; j < layer->map_count; j++)
		{
			index = i*layer->map_count + j;
			if (pconnection != NULL && !pconnection[index])
			{
				continue;
			}

			convn_valid(
				prev_layer->map[i].data, prev_layer->map_w, prev_layer->map_h,
				layer->map[j].error, layer->map_w, layer->map_h,
				layer->kernel[index].dW, layer->kernel_w, layer->kernel_h);
		}
	}

	// db
	size = layer->map_w * layer->map_h;
	for (int i = 0; i < layer->map_count; i++)
	{
		double sum = 0.0;
		for (int k = 0; k < size; k++)
		{
			sum += layer->map[i].error[k];
		}
		layer->map[i].db += sum;
	}
}

void avg_pooling_bprop(Layer *layer, Layer *prev_layer)
{
	const double scale_factor = 0.25;
	int size = prev_layer->map_w * prev_layer->map_h;

	for (int i = 0; i < layer->map_count; i++)
	{
		kronecker(layer->map[i].error, layer->map_w, layer->map_h, prev_layer->map_common, prev_layer->map_w);

		// delta
		for (int k = 0; k < size; k++)
		{
			double delta = layer->kernel[i].W[0] * prev_layer->map_common[k];
			prev_layer->map[i].error[k] = delta * scale_factor * activation_func::dtan_h(prev_layer->map[i].data[k]);
		}

		// dW
		double sum = 0.0;
		for (int k = 0; k < size; k++)
		{
			sum += prev_layer->map[i].data[k] * prev_layer->map_common[k];
		}
		layer->kernel[i].dW[0] += sum * scale_factor;

		// db
		sum = 0.0;
		for (int k = 0; k < layer->map_w * layer->map_h; k++)
		{
			sum += layer->map[i].error[k];
		}
		layer->map[i].db += sum;
	}
}

void max_pooling_bprop(Layer *layer, Layer *prev_layer)
{
	int map_w = layer->map_w;
	int map_h = layer->map_h;
	int upmap_w = prev_layer->map_w;

	for (int k = 0; k < layer->map_count; k++)
	{
		// delta
		for (int i = 0; i < map_h; i++)
		{
			for (int j = 0; j < map_w; j++)
			{
				int row = 2*i, col = 2*j;
				double max_value = prev_layer->map[k].data[row*upmap_w + col];
				for (int n = 2*i; n < 2*(i + 1); n++)
				{
					for (int m = 2*j; m < 2*(j + 1); m++)
					{
						if (prev_layer->map[k].data[n*upmap_w + m] > max_value)
						{
							row = n;
							col = m;
							max_value = prev_layer->map[k].data[n*upmap_w + m];
						}
						else
						{
							prev_layer->map[k].error[n*upmap_w + m] = 0.0;
						}
					}
				}

				prev_layer->map[k].error[row*upmap_w + col] = layer->map[k].error[i*map_w + j] * activation_func::dtan_h(max_value);
			}
		}

		// dW
		// db
	}
}

void fully_connected_bprop(Layer *layer, Layer *prev_layer)
{
	// delta
	for (int i = 0; i < prev_layer->map_count; i++)
	{
		prev_layer->map[i].error[0] = 0.0;
		for (int j = 0; j < layer->map_count; j++)
		{
			prev_layer->map[i].error[0] += layer->map[j].error[0] * layer->kernel[i*layer->map_count + j].W[0];
		}
		prev_layer->map[i].error[0] *= activation_func::dtan_h(prev_layer->map[i].data[0]);
	}

	// dW
	for (int i = 0; i < prev_layer->map_count; i++)
	{
		for (int j = 0; j < layer->map_count; j++)
		{
			layer->kernel[i*layer->map_count + j].dW[0] += layer->map[j].error[0] * prev_layer->map[i].data[0];
		}
	}

	// db
	for (int i = 0; i < layer->map_count; i++)
	{
		layer->map[i].db += layer->map[i].error[0];
	}
}

void backward_propagation(double *label)
{
	for (int i = 0; i < output_layer.map_count; i++)
	{
		output_layer.map[i].error[0] = loss_func::dmse(output_layer.map[i].data[0], label[i]) * activation_func::dtan_h(output_layer.map[i].data[0]);
	}

	// Out-->C5
	fully_connected_bprop(&output_layer, &c5_conv_layer);

	// C5-->S4
	conv_bprop(&c5_conv_layer, &s4_pooling_layer, NULL);

	// S4-->C3
	max_pooling_bprop(&s4_pooling_layer, &c3_conv_layer);/*avg_pooling_bprop*/

	// C3-->S2
	conv_bprop(&c3_conv_layer, &s2_pooling_layer, connection_table);

	// S2-->C1
	max_pooling_bprop(&s2_pooling_layer, &c1_conv_layer);/*avg_pooling_bprop*/

	// C1-->In
	conv_bprop(&c1_conv_layer, &input_layer, NULL);
}
//*-------------------------------------------------------------------------------------------------------/

// 更新权值----------------------------------------------------------------------------------------------/
inline double gradient_descent(double W, double dW, double alpha, double lambda)
{
	return W - alpha * (dW + lambda * W);
}

void update_params(Layer *layer, double learning_rate)
{
	const double lambda = 0.0;// 0.0002;

	// W
	int size = layer->kernel_w*layer->kernel_h;
	for (int i = 0; i < layer->kernel_count; i++)
	{
		for (int k = 0; k < size; k++)
		{
			layer->kernel[i].W[k] = gradient_descent(layer->kernel[i].W[k], layer->kernel[i].dW[k] / batch_size, learning_rate, lambda);
		}
	}

	// b
	for (int i = 0; i < layer->map_count; i++)
	{
		layer->map[i].b = gradient_descent(layer->map[i].b, layer->map[i].db / batch_size, learning_rate, lambda);
	}
}

void update_weights(double learning_rate)
{
	update_params(&c1_conv_layer, learning_rate);   // C1
	update_params(&s2_pooling_layer, learning_rate);// S2
	update_params(&c3_conv_layer, learning_rate);   // C3
	update_params(&s4_pooling_layer, learning_rate);// S4
	update_params(&c5_conv_layer, learning_rate);   // C5
	update_params(&output_layer, learning_rate);    // Out
}
//*-------------------------------------------------------------------------------------------------------/

// 重置参数----------------------------------------------------------------------------------------------/
void reset_params(Layer *layer)
{
	int mem_size = layer->kernel_w * layer->kernel_h * sizeof(double);
	for (int i = 0; i < layer->kernel_count; i++)
	{
		memset(layer->kernel[i].dW, 0, mem_size);
	}

	for (int i = 0; i < layer->map_count; i++)
	{
		layer->map[i].db = 0.0;
	}
}

void reset_weights()
{
	reset_params(&c1_conv_layer);   // C1
	reset_params(&s2_pooling_layer);// S2
	reset_params(&c3_conv_layer);   // C3
	reset_params(&s4_pooling_layer);// S4
	reset_params(&c5_conv_layer);   // C5
	reset_params(&output_layer);    // Out
}
//*-------------------------------------------------------------------------------------------------------/

// 训练--------------------------------------------------------------------------------------------------/
void train(Sample *train_sample, double learning_rate)
{
	// 随机打乱样本顺序
	int i = 0, j = 0, t = 0;
	int *rand_perm = (int *)malloc(train_sample->sample_count * sizeof(int));
	for (i = 0; i < train_sample->sample_count; i++)
	{
		rand_perm[i] = i;
	}

	for (i = 0; i < train_sample->sample_count; i++) 
	{
		j = genrand_int31() % (train_sample->sample_count - i) + i;
		t = rand_perm[j];
		rand_perm[j] = rand_perm[i];
		rand_perm[i] = t;
	}

	// 迭代训练
	int batch_count = train_sample->sample_count / batch_size;
	int data_mem_size = train_sample->sample_w * train_sample->sample_h * sizeof(double);
	for (i = 0; i < batch_count; i++)
	{
		// 重置参数
		reset_weights();

		for (j = 0; j < batch_size; j++)
		{
			// 填充数据
			int index = i*batch_size + j;
			memcpy(input_layer.map[0].data, train_sample[rand_perm[index]].data, data_mem_size);

			// 前向/反向传播计算
			forward_propagation();
			backward_propagation(train_sample[rand_perm[index]].label);
		}

		// 更新权值
		update_weights(learning_rate);

		if (i % 1000 == 0)
		{
			printf("progress...%d/%d \n", i, batch_count);
		}
	}

	free(rand_perm);
	rand_perm = NULL;
}
//*-------------------------------------------------------------------------------------------------------/

// 识别--------------------------------------------------------------------------------------------------/
int find_index(double *label)
{
	int index = 0;
	double max_val = label[0];
	for (int i = 1; i < classes_count; i++)
	{
		if (label[i] > max_val)
		{
			max_val = label[i];
			index = i;
		}
	}

	return index;
}

int find_index(Layer *layer)
{
	int index = 0;
	double max_val = *(layer->map[0].data);
	for (int i = 1; i < layer->map_count; i++)
	{
		if (*(layer->map[i].data) > max_val)
		{
			max_val = *(layer->map[i].data);
			index = i;
		}
	}

	return index;
}

void predict(Sample *test_sample)
{
	int num_success = 0, predict = 0, actual = 0;
	int data_mem_size = test_sample->sample_w * test_sample->sample_h * sizeof(double);
	int *confusion_matrix = (int *)malloc(classes_count * classes_count * sizeof(int));
	memset(confusion_matrix, 0, classes_count * classes_count * sizeof(int));

	for (int i = 0; i < test_sample->sample_count; i++)
	{
		memcpy(input_layer.map[0].data, test_sample[i].data, data_mem_size);
		forward_propagation();

		predict = find_index(&output_layer);
		actual = find_index(test_sample[i].label);
		if (predict == actual)
		{
			num_success++;
		}

		confusion_matrix[predict*classes_count + actual]++;
	}// for

	printf("accuracy: %d/%d\n", num_success, test_sample->sample_count);
	printf("\n   *  ");
	for (int i = 0; i < classes_count; i++)
	{
		printf("%4d  ", i);
	}

	printf("\n");
	for (int i = 0; i < classes_count; i++)
	{
		printf("%4d  ", i);
		for (int j = 0; j < classes_count; j++)
		{
			printf("%4d  ", confusion_matrix[i*classes_count + j]);
		}
		printf("\n");
	}
	printf("\n");

	free(confusion_matrix);
	confusion_matrix = NULL;
}
//*-------------------------------------------------------------------------------------------------------/

int main()
{
	int kernel_w = 0, kernel_h = 0;
	double learning_rate =  0.01 * sqrt((double)batch_size);

	// 训练数据
	Sample *train_sample = (Sample *)malloc(train_sample_count*sizeof(Sample));
	memset(train_sample, 0, train_sample_count*sizeof(Sample));
	train_sample->sample_w = width;
	train_sample->sample_h = height;
	train_sample->sample_count = train_sample_count;
	read_mnist_data(train_sample, "../TestLeNet/mnist/train-images.idx3-ubyte");
	read_mnist_label(train_sample, "../TestLeNet/mnist/train-labels.idx1-ubyte");

	// 测试数据
	Sample *test_sample = (Sample *)malloc(test_sample_count*sizeof(Sample));
	memset(test_sample, 0, test_sample_count*sizeof(Sample));
	test_sample->sample_w = width;
	test_sample->sample_h = height;
	test_sample->sample_count = test_sample_count;
	read_mnist_data(test_sample, "../TestLeNet/mnist/t10k-images.idx3-ubyte");
	read_mnist_label(test_sample, "../TestLeNet/mnist/t10k-labels.idx1-ubyte");

	// 随机数
	init_genrand((unsigned long)time(NULL));

	// 输入层In
	kernel_w = 0;
	kernel_h = 0;
	init_layer(&input_layer, 0, 1, kernel_w, kernel_h, width, height, false);

	// 卷积层C1
	kernel_w = 5;
	kernel_h = 5;
	init_layer(&c1_conv_layer, 1, 6, kernel_w, kernel_h, input_layer.map_w - kernel_w + 1, input_layer.map_h - kernel_h + 1, false);

	// 采样层S2
	kernel_w = 1;
	kernel_h = 1;
	init_layer(&s2_pooling_layer, 1, 6, kernel_w, kernel_h, c1_conv_layer.map_w / 2, c1_conv_layer.map_h / 2, true);

	// 卷积层C3
	kernel_w = 5;
	kernel_h = 5;
	init_layer(&c3_conv_layer, 6, 16, kernel_w, kernel_h, s2_pooling_layer.map_w - kernel_w + 1, s2_pooling_layer.map_h - kernel_h + 1, false);

	// 采样层S4
	kernel_w = 1;
	kernel_h = 1;
	init_layer(&s4_pooling_layer, 1, 16, kernel_w, kernel_h, c3_conv_layer.map_w / 2, c3_conv_layer.map_h / 2, true);

	// 卷积层C5
	kernel_w = 5;
	kernel_h = 5;
	init_layer(&c5_conv_layer, 16, 120, kernel_w, kernel_h, s4_pooling_layer.map_w - kernel_w + 1, s4_pooling_layer.map_h - kernel_h + 1, false);

	// 输出层Out
	kernel_w = 1;
	kernel_h = 1;
	init_layer(&output_layer, 120, 10, kernel_w, kernel_h, 1, 1, false);

	// 训练及测试
	clock_t start_time = 0;
	const int epoch = 50;
	for (int i = 0; i < epoch; i++)
	{
		printf("train epoch is %d ************************************************\n", i + 1);
		start_time = clock();
		train(train_sample, learning_rate);
		printf("train time is....%f s\n", (double)(clock() - start_time) / CLOCKS_PER_SEC);

		start_time = clock();
		predict(test_sample);
		printf("predict time is....%f s\n\n", (double)(clock() - start_time) / CLOCKS_PER_SEC);

		learning_rate *= 0.85;
	}

	// 释放资源
	for (int i = 0; i < train_sample_count; i++)
	{
		free(train_sample[i].data);
		free(train_sample[i].label);
		train_sample[i].data = NULL;
		train_sample[i].label = NULL;
	}
	free(train_sample);

	for (int i = 0; i < test_sample_count; i++)
	{
		free(test_sample[i].data);
		free(test_sample[i].label);
		test_sample[i].data = NULL;
		test_sample[i].label = NULL;
	}
	free(test_sample);

	release_layer(&input_layer);
	release_layer(&c1_conv_layer);
	release_layer(&c3_conv_layer);
	release_layer(&c5_conv_layer);
	release_layer(&s2_pooling_layer);
	release_layer(&s4_pooling_layer);
	release_layer(&output_layer);

	system("pause");
	return 0;
}
