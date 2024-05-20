#pragma once
#include "DeviceBuffer.cu"
#include "Utilities.h"

/// <summary>
/// Finds the total number of faces for the icosphere quantization, 
/// based on a recursive subdivision depth.
/// </summary>
/// <param name="depth">The subdivision depth.</param>
/// <returns>The total number of faces.</returns>
uint32_t getIcosphereFacesCountAtDepth(uint32_t depth);

/// <summary>
/// Finds the index of the bin a normal should be quantized to.
/// </summary>
/// <param name="normal">The normal.</param>
/// <param name="depth">The subdivision depth of the icosphere.</param>
/// <returns>The bin index.</returns>
uint32_t getIntersectedIcosphereFaceIndex(glm::vec3 normal, uint32_t depth);

/// <summary>
/// Quantizes a collection of normals into icosphere bins.
/// </summary>
/// <param name="args">The dispatch arguments provider.</param>
/// <param name="normals">the device buffer holding normals.</param>
/// <param name="bins">The device buffer holding bins.</param>
/// <param name="depth">The subdivision depth of the icosphere.</param>
void quantizeNormals(
	const DispatchArgs& args,
	DeviceBuffer<glm::vec3>& normals,
	DeviceBuffer<uint32_t>& bins,
	uint32_t depth);

/// <summary>
/// Calculates the endpoints of normals for visualization.
/// </summary>
/// <param name="args">The dispatch arguments provider.</param>
/// <param name="normals">The device buffer holding normals directions.</param>
/// <param name="vertices">The device buffer holding evaluated vertices.</param>
/// <param name="count">The number of normals to process.</param>
void evaluateNormalsEndpoints(
	const DispatchArgs& args,
	DeviceBuffer<glm::vec3>& normals,
	glm::vec3* normalVertices,
	int count);

/// <summary>
/// Reconstruct normals based on quantization bins.
/// Normals' magnitude reflects bins' count.
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
	uint32_t threshold);

/// <summary>
/// Performs a reduction on device buffers. Uses ping pong bufers.
/// Data to be reduced is expected in devDataIn.
/// </summary>
/// <param name="args">The dispatch arguments provider.</param>
/// <param name="devDataIn">The pointer to the device buffer holding the data to be reduced.</param>
/// <param name="devDataOut">The pointer to the device buffer to be used for ping pong swaps.</param>
/// <param name="count">The size of the input data to be processed.</param>
/// <returns>The result of the reduction, the sum of all input data.</returns>
float reduce(const DispatchArgs& args, float* devDataIn, float* devDataOut, int count);

/// <summary>
/// Performs a reduction on device buffers, using ping pong bufers.
/// </summary>
/// <param name="args">The dispatch arguments provider.</param>
/// <param name="input">The device buffer holding the data to be reduced.</param>
/// <param name="output">The device buffer to be used for ping pong swaps.</param>
/// <returns>The result of the reduction, the sum of all input data.</returns>
float reduce(const DispatchArgs& args, DeviceBuffer<float>& input, DeviceBuffer<float>& output);

/// <summary>
/// Evaluates a rotation fitting the set of normals.
/// That rotation essentially fits the data with a plane,
/// which amounts to evaluating one basis axis.
/// </summary>
/// <param name="args">The dispatch arguments provider.</param>
/// <param name="normals">The device buffer holding normals.</param>
/// <param name="scoresIn">The primary device buffer used for reduction, content is irrelevant.</param>
/// <param name="scoresOut">The secondary device buffer used for reduction, content is irrelevant.</param>
/// <param name="count">The number of normals to process.</param>
/// <param name="iterations">The number of iterations.</param>
/// <returns>The evaluated rotation.</returns>
glm::quat evaluateRotationPlane(
	const DispatchArgs& args,
	DeviceBuffer<glm::vec3>& normals,
	DeviceBuffer<float>& scoresIn,
	DeviceBuffer<float>& scoresOut,
	size_t count,
	int iterations);

/// <summary>
/// Evaluates a rotation fitting the set of normals.
/// We takes as input a rotation fitting the data with a plane,
/// then proceed to estimate the remaining 2 axes.
/// </summary>
/// <param name="args">The dispatch arguments provider.</param>
/// <param name="normals">The device buffer holding normals.</param>
/// <param name="scoresIn">The primary device buffer used for reduction, content is irrelevant.</param>
/// <param name="scoresOut">The secondary device buffer used for reduction, content is irrelevant.</param>
/// <param name="count">The number of normals to process.</param>
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
	glm::quat initRotation);