#include <stdio.h>
#include <stdlib.h>
#include <iostream>

#include <glm/gtc/random.hpp >
#include <glm/gtc/quaternion.hpp>
#include <glm/gtc/matrix_access.hpp>
#include <glm/gtx/string_cast.hpp >
using namespace glm;

#include "DeviceBuffer.cu"
#include "Utilities.h"
#include "EstimateRotation.h"
#include "Tests.h"

constexpr int testMultiProcessorCount = 16;

float reduce(const std::vector<float>& data)
{
	float* devDataIn;
	float* devDataOut;

	cudaMalloc((void**)&devDataIn, data.size() * sizeof(float));
	cudaMalloc((void**)&devDataOut, data.size() * sizeof(float));

	cudaMemcpy(devDataIn, &data[0], data.size() * sizeof(float), cudaMemcpyHostToDevice);

	DispatchArgs dispatchArgs(testMultiProcessorCount);
	auto result = reduce(dispatchArgs, devDataIn, devDataOut, data.size());

	cudaFree(devDataIn);
	cudaFree(devDataOut);

	return result;
}

void testMinDistCombinatorialBasis(int runs)
{
	for (auto i = 0; i != runs; ++i)
	{
		auto rndRot = glm::mat3_cast(getRandomRotation());
		auto trs = getRandomAxesSwapFlipMatrix();
		auto trsRndRot = rndRot * trs;
		auto extractedTrs = getMinDistChangeOfBasis(trsRndRot, rndRot);
		auto basisDist = getDistanceBetweenRotationMatrices(
			reOrthogonalize(trsRndRot * extractedTrs), reOrthogonalize(rndRot));
		std::cout << "Test Basis Extraction, distance: " << basisDist << std::endl;
	}
}

void testReduce()
{
	auto data = std::vector<float>(1024);
	auto expectedResult = 0;
	for (auto i = 0; i != data.size(); ++i)
	{
		data[i] = i + 1;
		expectedResult += i + 1;
	}

	auto result = reduce(data);
	auto error = result - expectedResult;
	std::cout << "Test Reduce, Error: " << error << std::endl;
}

void testReduceWithDeviceBuffers()
{
	auto data = std::vector<float>(1024);
	auto expectedResult = 0;
	for (auto i = 0; i != data.size(); ++i)
	{
		data[i] = i + 1;
		expectedResult += i + 1;
	}

	auto bufferIn = DeviceBuffer<float>(data.size());
	auto bufferOut = DeviceBuffer<float>(data.size());

	bufferIn.copyFrom(data);

	DispatchArgs dispatchArgs(testMultiProcessorCount);
	auto result = reduce(dispatchArgs, bufferIn, bufferOut);
	auto error = result - expectedResult;
	std::cout << "Test Reduce (with Device Buffers), Error: " << error << std::endl;
}

void testQuantizationWithDeviceBuffers(const std::vector<glm::vec3>& normals)
{
	constexpr auto depth = 4u;
	const auto numBins = getIcosphereFacesCountAtDepth(depth);

	DeviceBuffer<glm::vec3> normalsBuffer(normals);
	DeviceBuffer<uint32_t> binsBuffer(numBins, true);
	DeviceBuffer<glm::vec3> selectedNormalsBuffer(numBins);

	DispatchArgs dispatchArgs(testMultiProcessorCount);
	quantizeNormals(dispatchArgs, normalsBuffer, binsBuffer, depth);

	cudaDeviceSynchronize();

	auto bins = std::vector<uint32_t>(numBins);
	binsBuffer.copyTo(bins);

	uint32_t totalBins = 0u;
	for (auto i = 0; i != bins.size(); ++i)
	{
		totalBins += bins[i];
	}
	std::cout << "Total Bins (with Device Buffers): " << totalBins << std::endl;

	auto numSelectedNormals = reconstructFilteredNormals(dispatchArgs, binsBuffer, selectedNormalsBuffer, depth, 1u);

	cudaDeviceSynchronize();

	auto selectedNormals = std::vector<glm::vec3>(numSelectedNormals);
	selectedNormalsBuffer.copyTo(selectedNormals);

	auto totalNormals = 0.0f;
	for (auto i = 0; i != selectedNormals.size(); ++i)
	{
		totalNormals += glm::length(selectedNormals[i]);
	}
	std::cout << "Total Normals (with Device Buffers): " << totalNormals << std::endl;
}

void runTests()
{
	std::vector<glm::vec3> normals(1024);
	auto rotation = getRandomRotation();
	sampleRandomizedNormals(normals, rotation, 0.0f);

	testMinDistCombinatorialBasis(12);
	testReduce();
	testReduceWithDeviceBuffers();
	testQuantizationWithDeviceBuffers(normals);
}
