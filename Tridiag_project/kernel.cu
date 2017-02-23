#include "cuda_runtime.h"
#include "cuda.h"
#include "device_launch_parameters.h"

#include <time.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <algorithm>
#include <vector>
#include <iostream>

#define CUDA_CALL(x) do { cudaError_t err = x; if (( err ) != cudaSuccess ){ \
printf ("Error \"%s\" at %s :%d \n" , cudaGetErrorString(err), \
__FILE__ , __LINE__ ) ; return err;\
}} while (0);

void deBoorMakeTridiag(std::vector<float> x, std::vector<float> y, float d0, float dn, std::vector<float> &a, std::vector<float> &b, std::vector<float> &c, std::vector<float> &r)
{
	std::vector<float> dX(x.size() - 1);
	std::vector<float> dY(y.size() - 1);
	for (int i = 0; i < dX.size(); i++)
	{
		dX[i] = x[i + 1] - x[i];
		dY[i] = y[i + 1] - y[i];
	}
	for (int i = 0; i < a.size(); i++)
	{
		a[i] = dX[i + 1];
		b[i] = 2 * (dX[i] + dX[i + 1]);
		c[i] = dX[i];
		r[i] = 3 * ((dX[i] / dX[i + 1]) * dY[i + 1] + (dX[i + 1] / dX[i]) * dY[i]);
	}
	r[0] -= a[0] * d0;
	r[r.size() - 1] -= c[c.size() - 1] * dn;
}

__global__ void LU_tridiag(float* a, float* b, float* c, float* r, int from, int to)
{
	for (int i = from + 1; i < to; i++)
	{
		a[i] = a[i] / b[i - 1];
		b[i] = b[i] - (a[i] * c[i - 1]);
		r[i] = r[i] - (a[i] * r[i - 1]);
	}
	r[to - 1] = r[to - 1] / b[to - 1];
	for (int i = to - 2; i >= from; i--)
	{
		r[i] = (r[i] - (c[i] * r[i + 1])) / b[i];
	}
}

__device__ void LU_tridiag_device(float* a, float* b, float* c, float* r, int from, int to)
{
	for (int i = from + 1; i < to; i++)
	{
		a[i] = a[i] / b[i - 1];
		b[i] = b[i] - (a[i] * c[i - 1]);
		r[i] = r[i] - (a[i] * r[i - 1]);
	}
	r[to - 1] = r[to - 1] / b[to - 1];
	for (int i = to - 2; i >= from; i--)
	{
		r[i] = (r[i] - (c[i] * r[i + 1])) / b[i];
	}
}

void LU_CPU(std::vector<float> a, std::vector<float> b, std::vector<float> c, std::vector<float> &r, int from, int to)
{
	for (int i = from + 1; i < to; i++)
	{
		a[i] = a[i] / b[i - 1];
		b[i] = b[i] - (a[i] * c[i - 1]);
		r[i] = r[i] - (a[i] * r[i - 1]);
	}
	r[to - 1] = r[to - 1] / b[to - 1];
	for (int i = to - 2; i >= from; i--)
	{
		r[i] = (r[i] - (c[i] * r[i + 1])) / b[i];
	}
}

__global__ void partitioning(float* a, float* b, float* c, float* r, float* Va, float* Vb, float* Vc, float* Vr, int* Vindex, int pLength, int size, int Vsize, int remainder) {
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	int i = idx * 2 + 1;
	int myLength = pLength;
	int j = idx * myLength;
	if (i == Vsize - 1) // if this is the last processor
	{
		myLength += remainder;
	}
	if (i < Vsize)
	{
		/*float *dev_a = new float[myLength];
		float *dev_b = new float[myLength];
		float *dev_c = new float[myLength];
		float *dev_r = new float[myLength];
		for (int k = 0; k < myLength; k++)
		{
			dev_a[k] = a[j + k];
			dev_b[k] = b[j + k];
			dev_c[k] = c[j + k];
			dev_r[k] = r[j + k];
		}*/
		/*float Vai = dev_a[1];
		float Vbi = dev_b[1];
		float Vci = dev_c[1];
		float Vri = dev_r[1];*/
		float Vai = a[j + 1];
		float Vbi = b[j + 1];
		float Vci = c[j + 1];
		float Vri = r[j + 1];
		Vindex[i - 1] = j;
		Vindex[i] = j + myLength - 1;
		int jInit = j;
		for (int k = 2; k < myLength; k++) /* && j < size*/
		{
			float alpha = Vbi / a[j + k];
			Vri -= alpha * r[j + k];
			Vbi = Vci - alpha * b[j + k];
			Vci = -alpha * c[j + k];
		}
		Va[i] = Vai;
		Vb[i] = Vbi;
		Vc[i] = Vci;
		Vr[i] = Vri;
		i--;
		Vai = a[j + myLength - 2];
		Vbi = b[j + myLength - 2];
		Vci = c[j + myLength - 2];
		Vri = r[j + myLength - 2];
		for (int k = myLength - 3; k >= 0; k--)
		{
			float beta = Vbi / c[j + k];
			Vri = Vri - beta * r[j + k];
			Vbi = Vai - beta * b[j + k];
			Vai = -beta * a[j + k];
		}
		Va[i] = Vai;
		Vb[i] = Vbi;
		Vc[i] = Vci;
		Vr[i] = Vri;
		/*delete[] dev_a;
		delete[] dev_b;
		delete[] dev_c;
		delete[] dev_r;*/
	}
}

__global__ void final_computations(float* a, float* b, float* c, float* r, float* Vr, int* Vindex, int Vsize)
{
	int i = (blockDim.x * blockIdx.x + threadIdx.x) * 2;
	if (i < Vsize)
	{
		/*int Vind = Vindex[i];
		int Vind1 = Vindex[i + 1];
		float Vri = Vr[i];
		float Vri1 = Vr[i + 1];*/

		r[Vindex[i]] = Vr[i];
		r[Vindex[i + 1]] = Vr[i+1];

		int idx1 = Vindex[i] + 1;
		r[idx1] -= a[idx1] * Vr[i];

		int idx2 = Vindex[i + 1] - 1;
		r[idx2] -= c[idx2] * Vr[i+1];

		LU_tridiag_device(a, b, c, r, idx1, idx2 + 1);
	}
}

cudaError_t austin_berndt_moulton(std::vector<float> a, std::vector<float> b, std::vector<float> c, std::vector<float> &r, int nOfParts)
{
	int Vsize = nOfParts * 2;
	std::vector<float> Va(Vsize);
	std::vector<float> Vb(Vsize);
	std::vector<float> Vc(Vsize);
	std::vector<float> Vr(Vsize);
	std::vector<int> Vindex(Vsize);

	cudaEvent_t start, stop_malloc, stop_memcpy1, stop_partitioning, stop_seq, stop_final, stop_memcpy_final;
	float time1 = 0.0;
	float time2 = 0.0;
	float time3 = 0.0;
	float time4 = 0.0;
	float time5 = 0.0;
	float time6 = 0.0;
	cudaEventCreate(&start);
	cudaEventCreate(&stop_malloc);
	cudaEventCreate(&stop_memcpy1);
	cudaEventCreate(&stop_partitioning);
	cudaEventCreate(&stop_seq);
	cudaEventCreate(&stop_final);
	cudaEventCreate(&stop_memcpy_final);

	cudaEventRecord(start);
	cudaEventSynchronize(start);

	float *dev_a = 0; CUDA_CALL(cudaMalloc((void**)&dev_a, a.size() * sizeof(float)));
	float *dev_b = 0; CUDA_CALL(cudaMalloc((void**)&dev_b, b.size() * sizeof(float)));
	float *dev_c = 0; CUDA_CALL(cudaMalloc((void**)&dev_c, c.size() * sizeof(float)));
	float *dev_r = 0; CUDA_CALL(cudaMalloc((void**)&dev_r, r.size() * sizeof(float)));
	float *dev_Va = 0; CUDA_CALL(cudaMalloc((void**)&dev_Va, Vsize * sizeof(float)));
	float *dev_Vb = 0; CUDA_CALL(cudaMalloc((void**)&dev_Vb, Vsize * sizeof(float)));
	float *dev_Vc = 0; CUDA_CALL(cudaMalloc((void**)&dev_Vc, Vsize * sizeof(float)));
	float *dev_Vr = 0; CUDA_CALL(cudaMalloc((void**)&dev_Vr, Vsize * sizeof(float)));
	int *dev_Vidx = 0; CUDA_CALL(cudaMalloc((void**)&dev_Vidx, Vsize * sizeof(int)));

	cudaEventRecord(stop_malloc);
	cudaEventSynchronize(stop_malloc);
	cudaEventElapsedTime(&time1, start, stop_malloc);

	CUDA_CALL(cudaMemcpy(dev_a, &a[0], a.size() * sizeof(float), cudaMemcpyHostToDevice));
	CUDA_CALL(cudaMemcpy(dev_b, &b[0], b.size() * sizeof(float), cudaMemcpyHostToDevice));
	CUDA_CALL(cudaMemcpy(dev_c, &c[0], c.size() * sizeof(float), cudaMemcpyHostToDevice));
	CUDA_CALL(cudaMemcpy(dev_r, &r[0], r.size() * sizeof(float), cudaMemcpyHostToDevice));

	cudaEventRecord(stop_memcpy1);
	cudaEventSynchronize(stop_memcpy1);
	cudaEventElapsedTime(&time2, stop_malloc, stop_memcpy1);

	int pLength = r.size() / nOfParts;
	int remainder = r.size() - (pLength * nOfParts);
	int threadsPerBlock = 1024;
	int numBlocks = (nOfParts + threadsPerBlock - 1) / threadsPerBlock;

	CUDA_CALL(cudaSetDevice(0));

	partitioning<<<numBlocks, threadsPerBlock>>>(dev_a, dev_b, dev_c, dev_r, dev_Va, dev_Vb, dev_Vc, dev_Vr, dev_Vidx, pLength, r.size(), Vr.size(), remainder);
	CUDA_CALL(cudaGetLastError());
	CUDA_CALL(cudaDeviceSynchronize());

	cudaEventRecord(stop_partitioning);
	cudaEventSynchronize(stop_partitioning);
	cudaEventElapsedTime(&time3, stop_memcpy1, stop_partitioning);

	CUDA_CALL(cudaMemcpy(&Va[0], dev_Va, Vsize * sizeof(float), cudaMemcpyDeviceToHost));
	CUDA_CALL(cudaMemcpy(&Vb[0], dev_Vb, Vsize * sizeof(float), cudaMemcpyDeviceToHost));
	CUDA_CALL(cudaMemcpy(&Vc[0], dev_Vc, Vsize * sizeof(float), cudaMemcpyDeviceToHost));
	CUDA_CALL(cudaMemcpy(&Vr[0], dev_Vr, Vsize * sizeof(float), cudaMemcpyDeviceToHost));
	LU_CPU(Va, Vb, Vc, Vr, 0, Vsize);
	CUDA_CALL(cudaMemcpy(dev_Vr, &Vr[0], Vr.size() * sizeof(float), cudaMemcpyHostToDevice));
	
	/*LU_tridiag<<<1, 1>>>(dev_Va, dev_Vb, dev_Vc, dev_Vr, 0, Vsize);
	CUDA_CALL(cudaGetLastError());
	CUDA_CALL(cudaDeviceSynchronize());*/

	cudaEventRecord(stop_seq);
	cudaEventSynchronize(stop_seq);
	cudaEventElapsedTime(&time4, stop_partitioning, stop_seq);

	final_computations<<<numBlocks, threadsPerBlock>>>(dev_a, dev_b, dev_c, dev_r, dev_Vr, dev_Vidx, Vr.size());
	CUDA_CALL(cudaGetLastError());
	cudaError_t err = cudaDeviceSynchronize();
	if ((err) != cudaSuccess) {
		printf("Error \"%s\" at %s :%d \n", cudaGetErrorString(err), __FILE__, __LINE__);
	}

	cudaEventRecord(stop_final);
	cudaEventSynchronize(stop_final);
	cudaEventElapsedTime(&time5, stop_seq, stop_final);

	CUDA_CALL(cudaMemcpy(&r[0], dev_r, r.size() * sizeof(float), cudaMemcpyDeviceToHost));

	cudaEventRecord(stop_memcpy_final);
	cudaEventSynchronize(stop_memcpy_final);
	cudaEventElapsedTime(&time6, stop_final, stop_memcpy_final);

	std::cout << "malloc time: " << time1 << " ms" << std::endl;
	std::cout << "memcpy time: " << time2 << " ms" << std::endl;
	std::cout << "partit time: " << time3 << " ms" << std::endl;
	std::cout << "sequen time: " << time4 << " ms" << std::endl;
	std::cout << "fiinal time: " << time5 << " ms" << std::endl;
	std::cout << "rescpy time: " << time6 << " ms" << std::endl;
	std::cout << "sum time: " << time1+time2+time3+time4+time5+time6 << " ms" << std::endl;
	std::cout << "============================" << std::endl;

	return err;
}

void ABM_on_CPU(std::vector<float> a, std::vector<float> b, std::vector<float> c, std::vector<float> &r, int nOfParts) {
	int Vsize = nOfParts * 2;
	std::vector<float> Va(Vsize);
	std::vector<float> Vb(Vsize);
	std::vector<float> Vc(Vsize);
	std::vector<float> Vr(Vsize);
	std::vector<int> Vindex(Vsize);
	int j = 1;
	int pLength = b.size() / nOfParts;
	int remainder = b.size() - (pLength * nOfParts);
	for (int i = 0; i < Vb.size(); i += 2)
	{
		i++;
		if (i == Vb.size() - 1)
		{
			pLength += remainder;
		}
		Va[i] = a[j];
		Vb[i] = b[j];
		Vc[i] = c[j];
		Vr[i] = r[j];
		Vindex[i - 1] = j - 1;
		int jInit = j - 1;
		j++;
		for (int k = 0; k < pLength - 2 && j < b.size(); k++, j++)
		{
			float alpha = Vb[i] / a[j];
			Vr[i] -= alpha * r[j];
			Vb[i] = Vc[i] - alpha * b[j];
			Vc[i] = -alpha * c[j];
		}
		i--;
		Va[i] = a[j - 2];
		Vb[i] = b[j - 2];
		Vc[i] = c[j - 2];
		Vr[i] = r[j - 2];
		Vindex[i + 1] = j - 1;
		for (int k = j - 3; k >= jInit; k--)
		{
			float beta = Vb[i] / c[k];
			Vr[i] = Vr[i] - beta * r[k];
			Vb[i] = Va[i] - beta * b[k];
			Va[i] = -beta * a[k];
		}
		j++;
	}
	LU_CPU(Va, Vb, Vc, Vr, 0, Vsize);
	for (int i = 0; i < Vr.size(); i++)
	{
		r[Vindex[i]] = Vr[i];
	}
	for (int i = 0; i < Vr.size(); i += 2)
	{
		int idx1 = Vindex[i] + 1;
		r[idx1] -= a[idx1] * Vr[i];
		int idx2 = Vindex[i + 1] - 1;
		r[idx2] -= c[idx2] * Vr[i + 1];
		
		LU_CPU(a, b, c, r, idx1, idx2 + 1);
	}
}

int main()
{
	const int matrixSize = 500 * 1024;
	std::vector<float> a(matrixSize);
	std::vector<float> b(matrixSize);
	std::vector<float> c(matrixSize);
	std::vector<float> r(matrixSize);
	//float d1 = 1, dr = -1;
	//float x1 = -4, xr = 4;
	//std::vector<float> X(matrixSize + 2);
	//std::vector<float> F(matrixSize + 2);
	//float h = (xr - x1) / (X.size() - 1);
	//// Data X, F:
	//X[0] = x1;
	//F[0] = 1 / (1 + 4 * X[0] * X[0]);
	//for (int i = 1; i < X.size(); i++)
	//{
	//	X[i] = X[i - 1] + h; F[i] = 1 / (1 + 4 * X[i] * X[i]);
	//}
	//deBoorMakeTridiag(X, F, d1, dr, a, b, c, r);
	srand(time(NULL));
	for (size_t i = 0; i < matrixSize; i++)
	{
		a[i] = rand() % 10 + 1;
		c[i] = rand() % 10 + 1;
		b[i] = a[i] + c[i] + 1 + rand() % 10; // musi byt diagonalne dominantna
		r[i] = rand() % 100;
	}
	std::vector<float> a2(matrixSize);
	std::vector<float> b2(matrixSize);
	std::vector<float> c2(matrixSize);
	std::vector<float> r2(matrixSize);
	a2 = a;
	b2 = b;
	c2 = c;
	r2 = r;

	cudaEvent_t start, stop_CPU, stop_GPU;
	float time1 = 0.0;
	float time2 = 0.0;
	cudaEventCreate(&start);
	cudaEventCreate(&stop_CPU);
	cudaEventCreate(&stop_GPU);

	cudaEventRecord(start);
	cudaEventSynchronize(start);

	// computing on CPU
	LU_CPU(a2, b2, c2, r2, 0, r.size());
	//ABM_on_CPU(a2, b2, c2, r2, 1024);
	
	cudaEventRecord(stop_CPU);
	cudaEventSynchronize(stop_CPU);
	cudaEventElapsedTime(&time1, start, stop_CPU);

	// computing on GPU
    cudaError_t cudaStatus = austin_berndt_moulton(a, b, c, r, 1024);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "GPU computing failed!\n");
        return 1;
    }

	cudaEventRecord(stop_GPU);
	cudaEventSynchronize(stop_GPU);
	cudaEventElapsedTime(&time2, stop_CPU, stop_GPU);

	std::cout << "CPU time: " << time1 <<" ms" << std::endl;
	std::cout << "my GPU time: " << time2 << " ms" << std::endl << std::endl;
	// std::cout.precision(15);
	for (int i = 0; i < r.size(); i++)
	{
		float diff = r[i] - r2[i];
		if (diff > 0.00000000000001) { // 10^-15
			std::cout << "BACHA! rozdiel v " << i << " je presne " << diff << std::endl;
		}
	}
	/*std::cout << "R1: ";
	for (int i = 0; i < r.size(); i++)
	{
		std::cout << r[i] << ", ";
	}
	std::cout << std::endl << "R2: ";
	for (int i = 0; i < r.size(); i++)
	{
		std::cout << r2[i] << ", ";
	}
	std::cout << std::endl;*/
    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }

    return 0;
}