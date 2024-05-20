#include <math.h>
#include <vector>
#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>
#include <glm/gtc/matrix_access.hpp>
#include <glm/gtc/constants.hpp>
#include "EstimateRotation.h"

constexpr float X = .525731112119133606f;
constexpr float Z = .850650808352039932f;
constexpr float N = 0.f;

/// <summary>
/// Icosahedron vertices.
/// </summary>
__device__
constexpr float verticesLookup[] =
{
	-X,N,Z,
	X,N,Z,
	-X,N,-Z,
	X,N,-Z,
	N,Z,X,
	N,Z,-X,
	N,-Z,X,
	N,-Z,-X,
	Z,X,N,
	-Z,X, N,
	Z,-X, N,
	-Z,-X, N
};

__device__ __host__
glm::vec3 vertexAt(int index)
{
	return glm::vec3(
		verticesLookup[index * 3],
		verticesLookup[index * 3 + 1],
		verticesLookup[index * 3 + 2]);
}

/// <summary>
/// Icosahedron faces.
/// </summary>
__device__
constexpr int trianglesLookup[] =
{
	0,4,1,
	0,9,4,
	9,5,4,
	4,5,8,
	4,8,1,
	8,10,1,
	8,3,10,
	5,3,8,
	5,2,3,
	2,7,3,
	7,10,3,
	7,6,10,
	7,11,6,
	11,0,6,
	0,1,6,
	6,1,10,
	9,0,11,
	9,11,2,
	9,2,5,
	7,2,11
};

__device__ __host__
glm::ivec3 triangleAt(int index)
{
	return glm::ivec3(
		trianglesLookup[index * 3],
		trianglesLookup[index * 3 + 1],
		trianglesLookup[index * 3 + 2]);
}

/// <summary>
/// A simple struct holding the 3 vertices of a triangular face.
/// </summary>
struct Face
{
	glm::vec3 a;
	glm::vec3 b;
	glm::vec3 c;
};

/// <summary>
/// Number of icosahedron faces.
/// </summary>
constexpr unsigned int icosahedronFacesCount = 20u;

/// <summary>
/// Finds the total number of faces for the icosphere quantization, 
/// based on a recursive subdivision depth.
/// </summary>
/// <param name="depth">The subdivision depth.</param>
/// <returns>The total number of faces.</returns>
uint32_t getIcosphereFacesCountAtDepth(uint32_t depth)
{
	// icosahedronFacesCount then 2 more bits set to 1 per subdivision level.
	return (icosahedronFacesCount << (depth * 2));
}

/// <summary>
/// Finds whether a normal intersects a face.
/// </summary>
/// <remarks>
/// To determine if a normal intersects a face,
/// we check whether it lies on the right side of each of the 3 planes
/// defined by the icosphere center and 2 adjacent face vertices.
/// </remarks>
/// <param name="face">The face to test against.</param>
/// <param name="normal">The normal.</param>
/// <returns>True if there if an intersection, false otherwise.</returns>
__device__ __host__
bool normalIntersectsFace(Face face, glm::vec3 normal)
{
	auto a = glm::cross(face.c, face.b);
	auto b = glm::cross(face.a, face.c);
	auto c = glm::cross(face.b, face.a);
	auto an = glm::dot(a, normal);
	auto bn = glm::dot(b, normal);
	auto cn = glm::dot(c, normal);
	return an >= 0 && bn >= 0 && cn >= 0;
}

/// <summary>
/// Finds the index of the icosahedron face the normal intersects.
/// </summary>
/// <remarks>
/// The normal necessarily intersects a face, as there are no holes in the icosphere.
/// </remarks>
/// <param name="normal">The normal.</param>
/// <param name="index">The index of the intersected face.</param>
/// <returns>The intersected face.</returns>
__device__ __host__
Face findIntersectedIcosahedronFace(glm::vec3 normal, uint32_t& index)
{
	for (auto i = 0; i != icosahedronFacesCount; ++i)
	{
		auto indices = triangleAt(i);
		auto face = Face{ vertexAt(indices.x), vertexAt(indices.y), vertexAt(indices.z) };
		if (normalIntersectsFace(face, normal))
		{
			index = i;
			return face;
		}
	}

	// Should never happen. 
	// Normal has to hit a face regardless of direction.
	index = 0;
	return Face();
}

/// <summary>
/// Finds the subface of a face at a given index.
/// </summary>
/// <remarks>
/// There are 4 subfaces per face.
/// </remarks>
/// <param name="face">The original face.</param>
/// <param name="index">The index of the subface.</param>
/// <returns>The subface.</returns>
__device__ __host__
Face getSubFace(Face face, uint32_t index)
{
	assert(index < 4u);

	// Normalization puts the new vertex on the unit sphere surface.
	switch (index)
	{
	case 0: return Face{
		face.a,
		glm::normalize((face.a + face.b) * 0.5f),
		glm::normalize((face.c + face.a) * 0.5f) };
	case 1: return Face{
		glm::normalize((face.a + face.b) * 0.5f),
		face.b,
		glm::normalize((face.b + face.c) * 0.5f) };
	case 2: return Face{
		glm::normalize((face.a + face.b) * 0.5f),
		glm::normalize((face.b + face.c) * 0.5f),
		glm::normalize((face.c + face.a) * 0.5f) };
	case 3: return Face{
		glm::normalize((face.a + face.c) * 0.5f),
		glm::normalize((face.b + face.c) * 0.5f),
		face.c };
	}

	// Never happens, appease compiler.
	return Face();
}

/// <summary>
/// Finds the index of the face a normal should be quantized to.
/// </summary>
/// <param name="normal">The normal.</param>
/// <param name="depth">The subdivision depth of the icosphere.</param>
/// <returns>The face (bin) index.</returns>
__device__ __host__
uint32_t getIntersectedIcosphereFaceIndex(glm::vec3 normal, uint32_t depth)
{
	// First we look for the intersected face at depth 0.
	uint32_t index;
	auto face = findIntersectedIcosahedronFace(normal, index);

	// The we recursively look for intersected subfaces.
	for (auto i = 1; i != depth + 1; ++i)
	{
		// We use 2 bits per subdivsion levels.
		// As there are 4 subfaces per face.
		index <<= 2;

		// A face is subdivided by being sliced along 3 planes.
		// A vertex is added at the middle of each edge of the original face.
		auto ab = glm::normalize((face.a + face.b) * 0.5f);
		auto bc = glm::normalize((face.b + face.c) * 0.5f);
		auto ca = glm::normalize((face.c + face.a) * 0.5f);

		// There are 3 subfaces that can be identified by testing against one plane.
		// Those are the "outer" subfaces.
		auto a = glm::cross(ca, ab);
		auto an = glm::dot(a, normal);

		if (an >= 0)
		{
			face = Face{ face.a, ab, ca };
			continue;
		}

		auto b = glm::cross(ab, bc);
		auto bn = glm::dot(b, normal);

		if (bn >= 0)
		{
			index |= 1u;
			face = Face{ ab, b, bc };
			continue;
		}

		auto c = glm::cross(bc, ca);
		auto cn = glm::dot(c, normal);

		if (cn >= 0)
		{
			index |= 3u;
			face = Face{ ca, bc, c };
			continue;
		}

		// We have tested outer faces,
		// If we made it this far we necessarily are in the central subface.
		index |= 2u;
		face = Face{ ab, bc, ca };
	}

	return index;
}

/// <summary>
/// Finds the center of an icosphere face based on its index and the subdivision depth.
/// </summary>
/// <param name="index">The face (bin) index.</param>
/// <param name="depth">The icosphere subdivision depth.</param>
/// <returns>The center of the face.</returns>
__device__ __host__
glm::vec3 getIcosphereFaceCenter(uint32_t index, uint32_t depth)
{
	// Start by finding the face at depth 0.
	// Its index is stored in the first 5 (relevant) bits.
	auto faceIndex = index >> (2u * depth);
	auto faceIndices = triangleAt(faceIndex);
	auto face = Face{ vertexAt(faceIndices.x), vertexAt(faceIndices.y), vertexAt(faceIndices.z) };

	// Recursively look for subfaces.
	for (auto i = 1; i != depth + 1; ++i)
	{
		// Isolate the 2 bits describing the subface index at the current depth.
		auto subfaceIndex = (index >> ((depth - i) * 2)) & 3u;
		face = getSubFace(face, subfaceIndex);
	}

	// Normalize to put the vertex on the sphere surface.
	return glm::normalize((face.a + face.b + face.c) / 3.0f);
}

__global__
void quantizeKernel(glm::vec3* normals, uint32_t numNormals, uint32_t* bins, uint32_t depth)
{
	auto index = threadIdx.x + blockIdx.x * blockDim.x;
	if (index < numNormals)
	{
		auto binIndex = getIntersectedIcosphereFaceIndex(normals[index], depth);
		// Use atomic increment. Multiple threads may attempt to poke the same bin simultaneously.
		atomicAdd(&bins[binIndex], 1u);
	}
}

__global__
void filterBinsKernel(
	uint32_t* bins,
	glm::vec3* selectedNormals,
	uint32_t numBins,
	uint32_t threshold,
	uint32_t* writeIndex,
	uint32_t depth)
{
	auto index = threadIdx.x + blockIdx.x * blockDim.x;
	if (index < numBins)
	{
		auto bin = bins[index];
		if (bin >= threshold)
		{
			// Atomic operation to increment index and append to the list of selected normals.
			// A selected normal is one whose bin count passes the threshold test.
			auto insertIndex = atomicAdd(writeIndex, 1u);;
			// Reconstruct normal from bin index and count.
			// Index gives us direction, count gives us magnitude.
			auto normal = getIcosphereFaceCenter(index, depth) * (float)bin;
			selectedNormals[insertIndex] = normal;
		}
	}
}

/// <summary>
/// Quantizes a collection of normals into icosphere bins.
/// </summary>
/// <param name="args">The dispatch arguments provider.</param>
/// <param name="devNormals">The pointer to device normals buffer.</param>
/// <param name="devBins">The pointer to device bins buffer.</param>
/// <param name="count">Number of normals to process.</param>
/// <param name="depth">The subdivision depth of the icosphere.</param>
void quantizeNormals(
	const DispatchArgs& args,
	glm::vec3* devNormals,
	uint32_t* devBins,
	uint32_t count,
	uint32_t depth)
{
	auto numBins = getIcosphereFacesCountAtDepth(depth);
	dim3 dimBlock, dimGrid;
	args.getDims(count, dimBlock, dimGrid);
	quantizeKernel << <dimBlock, dimGrid >> > (devNormals, count, devBins, depth);
}

/// <summary>
/// Quantizes a collection of normals into icosphere bins.
/// </summary>
/// <param name="args">The dispatch arguments provider.</param>
/// <param name="normals">The device buffer holding normals.</param>
/// <param name="bins">The device buffers holding bins.</param>
/// <param name="depth">The subdivision depth of the icosphere.</param>
void quantizeNormals(
	const DispatchArgs& args,
	DeviceBuffer<glm::vec3>& normals,
	DeviceBuffer<uint32_t>& bins,
	uint32_t depth)
{
	assert(bins.size() == getIcosphereFacesCountAtDepth(depth));
	quantizeNormals(args, normals.getPtr(), bins.getPtr(), normals.size(), depth);
}

/// <summary>
/// Reconstruct normals based on quantization bins.
/// Normals' magnitude reflects bins count.
/// </summary>
/// <param name="args">The dispatch arguments provider.</param>
/// <param name="devBins">The pointer to device bins buffer.</param>
/// <param name="devNormals"The pointer to device reconstructed normals buffer.></param>
/// <param name="depth">The subdivision depth of the icosphere.</param>
/// <param name="threshold">Threshold above which a bin should be for a corresponding normal to be reconstructed.</param>
/// <returns>The number of reconstructed normals.</returns>
int reconstructFilteredNormals(
	const DispatchArgs& args,
	uint32_t* devBins,
	glm::vec3* devNormals,
	uint32_t depth,
	uint32_t threshold)
{
	auto numBins = getIcosphereFacesCountAtDepth(depth);
	int initialCounterValue = 0;
	uint32_t* devWriteIndex;
	cudaMalloc((void**)&devWriteIndex, sizeof(uint32_t));
	cudaMemcpy(devWriteIndex, &initialCounterValue, sizeof(uint32_t), cudaMemcpyHostToDevice);

	dim3 dimBlock, dimGrid;
	args.getDims(numBins, dimBlock, dimGrid);
	filterBinsKernel << <dimBlock, dimGrid >> > (devBins, devNormals, numBins, threshold, devWriteIndex, depth);

	cudaDeviceSynchronize();

	// Copy bins to CPU.
	int numSelectedNormals;
	cudaMemcpy(&numSelectedNormals, devWriteIndex, sizeof(int), cudaMemcpyDeviceToHost);
	return numSelectedNormals;
}

/// <summary>
/// Reconstruct normals based on quantization bins.
/// Normals' magnitude reflects bins count.
/// </summary>
/// <param name="args">The dispatch arguments provider.</param>
/// <param name="bins">The device buffer holding quantization bins.</param>
/// <param name="normals">The device buffer holding reconstructed normals.</param>
/// <param name="depth">The subdivision depth of the icosphere.</param>
/// <param name="threshold">The threshold above which a bin should be for a corresponding normal to be reconstructed.</param>
/// <returns>The number of reconstructed normals.</returns>
int reconstructFilteredNormals(
	const DispatchArgs& args,
	DeviceBuffer<uint32_t>& bins,
	DeviceBuffer<glm::vec3>& normals,
	uint32_t depth,
	uint32_t threshold)
{
	// At the very most we have one normal per bin.
	assert(bins.size() == getIcosphereFacesCountAtDepth(depth));
	normals.resizeIfNeeded(bins.size());
	return reconstructFilteredNormals(args, bins.getPtr(), normals.getPtr(), depth, threshold);
}

__device__
float square(float x)
{
	return x * x;
}

__device__
float squareLength(const glm::vec3& x)
{
	return glm::dot(x, x);
}

// Calculates the score of normals with respect to a basis (rotation).
// Normal's scores depend on their alignment with basis vectors. 
// A normal aligned with the basis can only be aligned with one of its vector, hence the use of max().
__global__
void evaluateRotationScoresKernel(glm::vec3* normals, float* scores, uint32_t count, glm::mat3 basis)
{
	auto index = threadIdx.x + blockIdx.x * blockDim.x;
	if (index < count)
	{
		auto normal = normals[index];
		auto x = square(glm::dot(normal, glm::column(basis, 0)));
		auto y = square(glm::dot(normal, glm::column(basis, 1)));
		auto z = square(glm::dot(normal, glm::column(basis, 2)));
		scores[index] = glm::max(x, glm::max(y, z));
	}
}

// Calculates the score of normals with respect to a basis vector.
// The higher the score the more the normal is aligned with the basis axis or lies on a plane perpendicular to it.
__global__
void evaluatePlaneScoresKernel(glm::vec3* normals, float* scores, uint32_t count, glm::vec3 basis)
{
	auto index = threadIdx.x + blockIdx.x * blockDim.x;
	if (index < count)
	{
		auto normal = normals[index];
		auto proj = glm::dot(basis, normal);
		auto projectDirOnNormal = basis * proj; // Alignment with the plane's normal.
		auto projectDirOnPlane = normal - projectDirOnNormal; // Alignment with the plane itself.
		// We then use max() as both alignments are valuable.
		scores[index] = glm::max(squareLength(projectDirOnNormal), squareLength(projectDirOnPlane));
	}
}

// Reduction (sum) kernel.
// See https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf
__global__
void reduceKernel(float* input, float* output, uint32_t count)
{
	extern __shared__ float sData[];

	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

	// First copy to shared memory.
	sData[tid] = i < count ? input[i] : 0.0f;
	__syncthreads();

	for (unsigned int s = blockDim.x / 2u; s > 0u; s >>= 1u)
	{
		if (tid < s)
		{
			sData[tid] += sData[tid + s];
		}
		__syncthreads();
	}

	// Write result to output.
	// Reduction of shared memory, one entry per block.
	if (tid == 0u)
	{
		output[blockIdx.x] = sData[0];
	}
}

/// <summary>
/// Performs a reduction on device buffers. Uses ping pong bufers.
/// Data to be reduced is expected in devDataIn.
/// </summary>
/// <param name="args">The dispatch arguments provider.</param>
/// <param name="devDataIn">The pointer to the device buffer holding the data to be reduced.</param>
/// <param name="devDataOut">The pointer to the device buffer to be used for ping pong swaps.</param>
/// <param name="count">The size of the input data to be processed.</param>
/// <returns>The result of the reduction, the sum of all input data.</returns>
float reduce(const DispatchArgs& args, float* devDataIn, float* devDataOut, int count)
{
	dim3 dimBlock, dimGrid;
	args.getDims(count, dimBlock, dimGrid);

	for (;;)
	{
		reduceKernel << <dimBlock, dimGrid >> > (devDataIn, devDataOut, count);

		count = dimBlock.x;
		dimGrid = dimBlock;

		if (dimGrid.x == 1)
		{
			break;
		}

		dimBlock.x = (int)glm::ceil(dimBlock.x / (float)dimGrid.x);

		std::swap(devDataIn, devDataOut);
	}

	cudaDeviceSynchronize();

	// Read reduced score at index 0.
	float result;
	cudaMemcpy(&result, devDataOut, sizeof(float), cudaMemcpyDeviceToHost);

	return result;
}

// Calculates the endpoints of normals for visualization.
__global__
void evaluateNormalsEndpointsKernel(glm::vec3* normals, glm::vec3* vertices, uint32_t count)
{
	auto index = threadIdx.x + blockIdx.x * blockDim.x;
	if (index < count)
	{
		auto normal = normals[index];
		auto origin = glm::normalize(normal);
		vertices[index * 2] = origin;
		vertices[index * 2 + 1] = origin + normal;
	}
}

/// <summary>
/// Calculates the endpoints of normals for visualization.
/// </summary>
/// <param name="args">The dispatch arguments provider.</param>
/// <param name="normals">The device buffer holding normals directions.</param>
/// <param name="vertices">The device buffer holding evaluated vertices.</param>
/// <param name="count">The number of normals to process.</param>
void evaluateNormalsEndpoints(const DispatchArgs& args, DeviceBuffer<glm::vec3>& normals, glm::vec3* vertices, int count)
{
	dim3 dimBlock, dimGrid;
	args.getDims(count, dimBlock, dimGrid);
	evaluateNormalsEndpointsKernel << <dimBlock, dimGrid >> > (normals.getPtr(), vertices, count);
}

/// <summary>
/// Performs a reduction on device buffers, using ping pong bufers.
/// </summary>
/// <param name="args">The dispatch arguments provider.</param>
/// <param name="input">The device buffer holding the data to be reduced.</param>
/// <param name="output">The device buffer to be used for ping pong swaps.</param>
/// <returns>The result of the reduction, the sum of all input data.</returns>
float reduce(const DispatchArgs& args, DeviceBuffer<float>& input, DeviceBuffer<float>& output)
{
	return reduce(args, input.getPtr(), output.getPtr(), input.size());
}

constexpr int searchUpRotationStep0Length = 9;
const glm::vec2 searchUpRotationStep0[searchUpRotationStep0Length] =
{
	glm::vec2(0, 0),
	glm::vec2(0.25f, 0),
	glm::vec2(0.25f, 0.25f),
	glm::vec2(0.25f, 0.50f),
	glm::vec2(0.25f, 0.75f),
	glm::vec2(0.25f, 1.0f),
	glm::vec2(0.25f, 1.25f),
	glm::vec2(0.25f, 1.50f),
	glm::vec2(0.25f, 1.75f)
};

// Infer a quaternion from pitch and yaw coordinates.
glm::quat getPitchYawRotation(glm::vec2 pitchYaw)
{
	auto quatPitch = glm::angleAxis(pitchYaw.x, glm::vec3(1, 0, 0));
	auto quatYaw = glm::angleAxis(pitchYaw.y, glm::vec3(0, 1, 0));
	return quatYaw * quatPitch;
}

/// <summary>
/// Evaluates a rotation fitting the set of normals.
/// That rotation essentially fits the data with a plane,
/// which amounts to evaluating one basis axis.
/// </summary>
/// <param name="args">The dispatch arguments provider.</param>
/// <param name="devNormals">The pointer to a device buffer holding normals.</param>
/// <param name="devScoresIn">The pointer to the primary device buffer used for reduction, content is irrelevant.</param>
/// <param name="devScoresOut">The pointer to the secondary device buffer used for reduction, content is irrelevant.</param>
/// <param name="count">The number of normals to process.</param>
/// <param name="iterations">The number of iterations.</param>
/// <returns>The evaluated rotation.</returns>
glm::quat evaluateRotationPlane(
	const DispatchArgs& args,
	glm::vec3* devNormals,
	float* devScoresIn,
	float* devScoresOut,
	size_t count,
	int iterations)
{
	auto bestScore = 0.0f;
	glm::quat bestRotation;
	glm::vec2 basePitchYaw;

	dim3 dimBlock, dimGrid;
	args.getDims(count, dimBlock, dimGrid);

	for (auto i = 0; i != searchUpRotationStep0Length; ++i)
	{
		auto pitchYaw = searchUpRotationStep0[i] * glm::pi<float>();
		auto rotation = getPitchYawRotation(pitchYaw);

		auto basis = rotation * glm::vec3(0, 1, 0);

		// Evaluate scores.
		evaluatePlaneScoresKernel << <dimBlock, dimGrid >> > (devNormals, devScoresIn, count, basis);

		auto score = reduce(args, devScoresIn, devScoresOut, count);

		if (score > bestScore)
		{
			bestScore = score;
			bestRotation = rotation;
			basePitchYaw = pitchYaw;
		}
	}

	auto remainingSteps = glm::max(0, iterations - 1);
	auto bestPitchYaw = basePitchYaw;
	// At step 0 angular range is PI / 4, so we pick up at PI / 8.
	auto angularSpan = glm::pi<float>() * 0.125f;

	for (auto i = 0; i != remainingSteps; ++i)
	{
		for (auto yaw = -1; yaw != 2; ++yaw)
		{
			for (auto pitch = -1; pitch != 2; ++pitch)
			{
				// Previously evaluated.
				if (yaw == 0 && pitch == 0)
				{
					continue;
				}

				auto pitchYaw = basePitchYaw + glm::vec2(pitch, yaw) * angularSpan;
				auto rotation = getPitchYawRotation(pitchYaw);

				auto basis = rotation * glm::vec3(0, 1, 0);

				// Evaluate scores.
				evaluatePlaneScoresKernel << <dimBlock, dimGrid >> > (devNormals, devScoresIn, count, basis);

				auto score = reduce(args, devScoresIn, devScoresOut, count);

				if (score > bestScore)
				{
					bestScore = score;
					bestPitchYaw = pitchYaw;
					bestRotation = rotation;
				}
			}
		}

		basePitchYaw = bestPitchYaw;

		angularSpan *= 0.5f;
	}

	return bestRotation;
}

/// <summary>
/// Evaluates a rotation fitting the set of normals.
/// That rotation takes as input a rotation fitting the data with a plane,
/// then proceeds to estimate the remaining 2 axes.
/// </summary>
/// <param name="args">The dispatch arguments provider.</param>
/// <param name="devNormals">The pointer to the device buffer holding normals.</param>
/// <param name="devScoresIn">The pointer to the primary device buffer used for reduction, content is irrelevant.</param>
/// <param name="devScoresOut">The pointer to the secondary device buffer used for reduction, content is irrelevant.</param>
/// <param name="count">The number of normals to process.</param>
/// <param name="iterations">The number of iterations.</param>
/// <param name="initRotation">The partial rotation evaluated using <see cref="evaluateRotationPlane"/>.</param>
/// <returns>The evaluated rotation.</returns>
glm::quat evaluateRotationBasis(
	const DispatchArgs& args,
	glm::vec3* devNormals,
	float* devScoresIn,
	float* devScoresOut,
	size_t count,
	int iterations,
	glm::quat initRotation)
{
	auto angularSpan = glm::pi<float>() * 0.25f;
	auto baseYaw = 0.0f;
	auto bestScore = 0.0f;
	float bestYaw;
	glm::quat bestRotation;

	dim3 dimBlock, dimGrid;
	args.getDims(count, dimBlock, dimGrid);

	for (auto i = 0; i != iterations; ++i)
	{
		for (auto j = -1; j != 2; ++j)
		{
			auto yaw = baseYaw + angularSpan * j;
			auto quatYaw = glm::angleAxis(yaw, glm::vec3(0, 1, 0));
			auto rotation = initRotation * quatYaw;
			auto rotationMat3 = glm::mat3_cast(rotation);

			// Evaluate scores.
			evaluateRotationScoresKernel << <dimBlock, dimGrid >> > (devNormals, devScoresIn, count, rotationMat3);

			auto score = reduce(args, devScoresIn, devScoresOut, count);

			if (score > bestScore)
			{
				bestScore = score;
				bestYaw = yaw;
				bestRotation = rotation;
			}
		}

		baseYaw = bestYaw;

		angularSpan *= 0.5f;
	}

	return bestRotation;
}

/// <summary>
/// Evaluates a rotation fitting the set of normals.
/// That rotation essentially fits the data with a plane,
/// which amounts to evaluating one basis axis.
/// </summary>
/// <param name="args">The dispatch arguments provider.</param>
/// <param name="normals">The device buffer holding normals.</param>
/// <param name="scoresIn">the primary device buffer used for reduction, content is irrelevant.</param>
/// <param name="scoresOut">The secondary device buffer used for reduction, content is irrelevant.</param>
/// <param name="size">The number of normals to process.</param>
/// <param name="iterations">The number of iterations.</param>
/// <returns>The evaluated rotation.</returns>
glm::quat evaluateRotationPlane(
	const DispatchArgs& args,
	DeviceBuffer<glm::vec3>& normals,
	DeviceBuffer<float>& scoresIn,
	DeviceBuffer<float>& scoresOut,
	size_t count,
	int iterations)
{
	scoresIn.resizeIfNeeded(normals.size());
	scoresOut.resizeIfNeeded(normals.size());
	return evaluateRotationPlane(args, normals.getPtr(), scoresIn.getPtr(), scoresOut.getPtr(), count, iterations);
}

/// <summary>
/// Evaluates a rotation fitting the set of normals.
/// That rotation takes as input a rotation fitting the data with a plane,
/// then proceeds to estimate the remaining 2 axes.
/// </summary>
/// <param name="args">The dispatch arguments provider.</param>
/// <param name="normals">The device buffer holding normals.</param>
/// <param name="scoresIn">The primary device buffer used for reduction, content is irrelevant.</param>
/// <param name="scoresOut">The secondary device buffer used for reduction, content is irrelevant.</param>
/// <param name="size">The number of normals to process.</param>
/// <param name="iterations">The number of iterations.</param>
/// <param name="initRotation">The partial rotation evaluated using <see cref="evaluateRotationPlane"/>.</param>
/// <returns>The evaluated rotation.</returns>
glm::quat evaluateRotationBasis(
	const DispatchArgs& args,
	DeviceBuffer<glm::vec3>& normals,
	DeviceBuffer<float>& scoresIn,
	DeviceBuffer<float>& scoresOut,
	size_t count,
	int iterations,
	glm::quat initRotation)
{
	scoresIn.resizeIfNeeded(normals.size());
	scoresOut.resizeIfNeeded(normals.size());
	return evaluateRotationBasis(args, normals.getPtr(), scoresIn.getPtr(), scoresOut.getPtr(), count, iterations, initRotation);
}
