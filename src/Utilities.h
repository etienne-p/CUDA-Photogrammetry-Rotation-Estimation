#pragma once

#include <GL/glew.h>
#include <glm/glm.hpp>
#include <vector>
#include <string>
#include <vector_types.h>

std::vector<GLfloat> loadVerticesFromPly(std::string fileName, glm::vec3 center);

GLuint compileShadersProgram(const char* vertexShader, const char* fragmentShader);

enum class CubemapFace
{
	XPositive = 0,
	XNegative = 1,
	YPositive = 2,
	YNegative = 3,
	ZPositive = 4,
	ZNegative = 5
};

constexpr int facesPerCube = 6;

glm::quat getRandomRotation(float angularRange);

glm::quat getRandomRotation();

glm::mat3 getRandomAxesSwapFlipMatrix();

glm::vec3 toVec3(CubemapFace face);

glm::mat3 reOrthogonalize(glm::mat3 r);

float getDistanceBetweenRotationMatrices(glm::mat3 a, glm::mat3 b);

glm::mat3 getMinDistChangeOfBasis(glm::mat3 a, glm::mat3 b);

void sampleRandomizedNormals(
	std::vector<glm::vec3>& normals,
	glm::quat rotation, float gaussianDeviationRadians);

struct DispatchArgs
{
private:
	int m_MultiProcessorCount;
public:
	DispatchArgs(int multiProcessorCount) : m_MultiProcessorCount(multiProcessorCount) {}
	void getDims(int count, dim3& dimBlock, dim3& dimGrid) const
	{
		auto idealOccupancy = (int)glm::ceil(count / (float)m_MultiProcessorCount);

		// Next power of 2 after idealOccupancy.
		auto threadPerBlock = 1;
		while (threadPerBlock < idealOccupancy)
		{
			threadPerBlock *= 2;
		}

		// If we exceed thread count limit, roll back.
		constexpr int maxThreadPerBlock = 1024;

		while (threadPerBlock > maxThreadPerBlock)
		{
			threadPerBlock /= 2;
		}
		auto numBlocks = (count + threadPerBlock - 1) / threadPerBlock;

		dimGrid = dim3(threadPerBlock);
		dimBlock = dim3(numBlocks);

		assert(dimGrid.x * dimBlock.x >= count);
		assert(dimGrid.x * (dimBlock.x - 1) <= count);
	}
};