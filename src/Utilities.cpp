#include <GL/glew.h>

#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>
#include <glm/gtx/quaternion.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/matrix_access.hpp>
#include <glm/gtc/random.hpp >
using namespace glm;

#include "Utilities.h"

GLuint compileShadersProgram(const char* vertexShader, const char* fragmentShader) {

	// Create the shaders
	auto vertexShaderId = glCreateShader(GL_VERTEX_SHADER);
	auto fragmentShaderId = glCreateShader(GL_FRAGMENT_SHADER);

	GLint result = GL_FALSE;
	int infoLogLength;

	// Compile Vertex Shader
	glShaderSource(vertexShaderId, 1, &vertexShader, NULL);
	glCompileShader(vertexShaderId);

	// Check Vertex Shader
	glGetShaderiv(vertexShaderId, GL_COMPILE_STATUS, &result);
	glGetShaderiv(vertexShaderId, GL_INFO_LOG_LENGTH, &infoLogLength);

	if (infoLogLength > 0)
	{
		std::vector<char> VertexShaderErrorMessage(infoLogLength + 1);
		glGetShaderInfoLog(vertexShaderId, infoLogLength, NULL, &VertexShaderErrorMessage[0]);
		printf("%s\n", &VertexShaderErrorMessage[0]);
	}

	// Compile Fragment Shader
	glShaderSource(fragmentShaderId, 1, &fragmentShader, NULL);
	glCompileShader(fragmentShaderId);

	// Check Fragment Shader
	glGetShaderiv(fragmentShaderId, GL_COMPILE_STATUS, &result);
	glGetShaderiv(fragmentShaderId, GL_INFO_LOG_LENGTH, &infoLogLength);

	if (infoLogLength > 0)
	{
		std::vector<char> FragmentShaderErrorMessage(infoLogLength + 1);
		glGetShaderInfoLog(fragmentShaderId, infoLogLength, NULL, &FragmentShaderErrorMessage[0]);
		printf("%s\n", &FragmentShaderErrorMessage[0]);
	}

	// Link the program
	auto programId = glCreateProgram();
	glAttachShader(programId, vertexShaderId);
	glAttachShader(programId, fragmentShaderId);
	glLinkProgram(programId);

	// Check the program
	glGetProgramiv(programId, GL_LINK_STATUS, &result);
	glGetProgramiv(programId, GL_INFO_LOG_LENGTH, &infoLogLength);

	if (infoLogLength > 0)
	{
		std::vector<char> ProgramErrorMessage(infoLogLength + 1);
		glGetProgramInfoLog(programId, infoLogLength, NULL, &ProgramErrorMessage[0]);
		printf("%s\n", &ProgramErrorMessage[0]);
	}

	glDetachShader(programId, vertexShaderId);
	glDetachShader(programId, fragmentShaderId);

	glDeleteShader(vertexShaderId);
	glDeleteShader(fragmentShaderId);

	return programId;
}

glm::vec3 toVec3(CubemapFace face)
{
	switch (face)
	{
	case CubemapFace::XPositive:
		return glm::vec3(1, 0, 0);
	case CubemapFace::XNegative:
		return glm::vec3(-1, 0, 0);
	case CubemapFace::YPositive:
		return glm::vec3(0, 1, 0);
	case CubemapFace::YNegative:
		return glm::vec3(0, -1, 0);
	case CubemapFace::ZPositive:
		return glm::vec3(0, 0, 1);
	case CubemapFace::ZNegative:
		return glm::vec3(0, 0, -1);
	}

	// Never happens, appease compiler.
	return glm::vec3();
}

glm::quat getRandomRotation(float angularRange)
{
	auto rotAxis = glm::sphericalRand(1.0f);
	auto rotAngle = glm::linearRand(-angularRange, angularRange);
	return glm::normalize(glm::angleAxis(rotAngle, rotAxis));
}

glm::quat getRandomGaussRotation(float deviation)
{
	auto rotAxis = glm::sphericalRand(1.0f);
	auto rotAngle = glm::gaussRand(0.0f, deviation);
	return glm::normalize(glm::angleAxis(rotAngle, rotAxis));
}

glm::quat getRandomRotation()
{
	return getRandomRotation(glm::pi<float>());
}

glm::mat3 getRandomAxesSwapFlipMatrix()
{
	auto m = glm::mat3(1.0f); // Identity.
	for (auto i = 0; i != 3; ++i)
	{
		// Randomly swap columns.
		if (glm::linearRand(0.0f, 1.0f) > 0.5f)
		{
			std::swap(m[i], m[(i + 1) % 3]);
		}

		// Randomly flip columns.
		if (glm::linearRand(0.0f, 1.0f) > 0.5f)
		{
			m[i] *= -1.0f;
		}
	}
	return m;
}

// See http://www.boris-belousov.net/2016/12/01/quat-dist/#using-rotation-matrices
float getDistanceBetweenRotationMatrices(glm::mat3 a, glm::mat3 b)
{
	auto diff = a * glm::inverse(b);
	auto trace = diff[0][0] + diff[1][1] + diff[2][2];
	auto cos = (trace - 1.0f) / 2.0f;
	// Due to imprecision it is possible to fall outside the [-1, 1] cos range.
	cos = glm::clamp(cos, -1.0f, 1.0f);
	return glm::acos(cos);
}

// This method is meant to re-orthogonalize a matrix that is not far from being orthogonal.
// Method described in Direction Cosine Matrix IMU: Theory, by William Premerlani and Paul Bizard
glm::mat3 reOrthogonalize(glm::mat3 r)
{
	// First we compute the dot product of the X and Y rows of the matrix,
	// which is supposed to be zero, so the result is a measure
	// of how much the X and Y rows are rotating toward each other.
	auto error = glm::dot(r[0], r[1]);
	// We apportion half of the error each to the X and Y rows, and approximately
	// rotate the X and Y rows in the opposite direction by cross coupling.
	auto xOrt = r[0] - (error / 2) * r[1];
	auto yOrt = r[1] - (error / 2) * r[0];
	// Adjust the Z row of the matrix to be orthogonal to the X and Y row.
	// The way we do that is to simply set the Z row to be the cross product of the X and Y rows.
	auto zOrt = glm::cross(xOrt, yOrt);
	// Last step, normalize orthogonal axes.
	return glm::mat3(
		glm::normalize(xOrt),
		glm::normalize(yOrt),
		glm::normalize(zOrt));
}

// a and b: orthogonal rotation matrices.
// Evaluates basis M so that dist(a ,b) > dist(a * M, b).
// M's axes are aligned with basis vectors.
// That is, it's swap & flips of basis vectors.
glm::mat3 getMinDistChangeOfBasis(glm::mat3 a, glm::mat3 b)
{
	// We expect othogonal rotation matrices, transpose is inverse.
	auto delta = glm::transpose(a) * b;

	// Possible axes indices permutations.
	const glm::ivec3 permutations[] = {
		glm::ivec3(0, 1, 2),
		glm::ivec3(0, 2, 1),
		glm::ivec3(1, 0, 2),
		glm::ivec3(1, 2, 0),
		glm::ivec3(2, 0, 1),
		glm::ivec3(2, 1, 0)
	};

	glm::ivec3 bestAxes;
	float bestScore = 0.0f;

	// First select the best permutation.
	for (auto i = 0; i != 6; ++i)
	{
		auto axes = permutations[i];
		// Think of this like a dot with the basis vector,
		// considering that dot(vec3(1, 0, 0), v) = v.x.
		auto score = glm::abs(delta[0][axes.x]) + glm::abs(delta[1][axes.y]) + glm::abs(delta[2][axes.z]);
		if (score > bestScore)
		{
			bestScore = score;
			bestAxes = axes;
		}
	}

	// The establish axes directions.
	auto result = glm::mat3(0.0f);
	result[0][bestAxes.x] = glm::sign(delta[0][bestAxes.x]);
	result[1][bestAxes.y] = glm::sign(delta[1][bestAxes.y]);
	result[2][bestAxes.z] = glm::sign(delta[2][bestAxes.z]);
	return result;
}

// Populates normals with axis aligned normals, in all directions.
// We apply a rotation, as well as a randomized rotation of a passed angular range.
void sampleRandomizedNormals(std::vector<glm::vec3>& normals, glm::quat rotation, float gaussianDeviationRadians)
{
	for (auto i = 0; i != normals.size(); ++i)
	{
		auto autoFaceIndex = glm::linearRand(0, 5);
		auto face = static_cast<CubemapFace>(autoFaceIndex);
		auto normal = toVec3(face);
		auto randomRotation = getRandomGaussRotation(gaussianDeviationRadians);
		auto transform = rotation * randomRotation;
		normals[i] = glm::rotate(transform, normal);
	}
}

