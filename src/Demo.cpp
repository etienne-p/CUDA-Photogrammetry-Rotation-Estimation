#include <stdio.h>
#include <vector>

#include <GL/glew.h>

#include <GLFW/glfw3.h>
GLFWwindow* window;

#include "cuda_gl_interop.h"

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/random.hpp >
#include <glm/gtc/quaternion.hpp>
#include <glm/gtx/quaternion.hpp>
#include <glm/gtc/matrix_access.hpp>
#include <glm/gtx/string_cast.hpp >
using namespace glm;

#include <AntTweakBar.h>

#include "DeviceBuffer.cu"
#include "Utilities.h"
#include "EstimateRotation.h"
#include "Tests.h"
#include <cuda_runtime_api.h>
#include <iostream>

#define STRINGIFY(A) #A

const char* vertexShader = STRINGIFY(
	#version 330 core                                                          \n
	layout(location = 0) in vec3 vertexPosition_modelspace;                    \n
	uniform mat4 ModelViewProjection;                                          \n
	void main()                                                                \n
	{ \n
		gl_Position = ModelViewProjection * vec4(vertexPosition_modelspace, 1); \n
	}                                                                 \n
);

const char* fragmentShader = STRINGIFY(
	#version 330 core            \n
	out vec3 color;              \n
	void main()                  \n
	{ \n
		color = vec3(1, 0, 0);   \n
	}                            \n
);

struct OrbitCamera
{
private:
	glm::vec3 m_CameraTarget;
	float m_Phi = 90.f;
	float m_Theta = 0;
	float m_Radius = 4;

public:
	void Update()
	{
		const float delta = 2.f;
		if (glfwGetKey(window, GLFW_KEY_RIGHT) == GLFW_PRESS)
		{
			m_Theta += delta;
		}
		if (glfwGetKey(window, GLFW_KEY_LEFT) == GLFW_PRESS)
		{
			m_Theta -= delta;
		}
		if (glfwGetKey(window, GLFW_KEY_UP) == GLFW_PRESS)
		{
			m_Phi += delta;
		}
		if (glfwGetKey(window, GLFW_KEY_DOWN) == GLFW_PRESS)
		{
			m_Phi -= delta;
		}

		m_Phi = glm::clamp(m_Phi, 1.f, 179.f);
	}

	glm::mat4 GetModelViewProjection()
	{
		glm::vec3 cameraPosition;
		cameraPosition.x = m_Radius * glm::sin(glm::radians(m_Phi)) * glm::cos(glm::radians(m_Theta));
		cameraPosition.y = m_Radius * glm::cos(glm::radians(m_Phi));
		cameraPosition.z = m_Radius * glm::sin(glm::radians(m_Phi)) * glm::sin(glm::radians(m_Theta));

		auto view = glm::lookAt(cameraPosition, m_CameraTarget, glm::vec3(0, 1, 0));
		auto projection = glm::perspective(glm::radians(60.0f), 1.f, 1.f, 200.0f);
		return projection * view;
	}
};

int main(void)
{
	// Initialize GLFW
	if (!glfwInit())
	{
		fprintf(stderr, "Failed to initialize GLFW\n");
		getchar();
		return -1;
	}

	glfwWindowHint(GLFW_SAMPLES, 4);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
	glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE); // To make macOS happy; should not be needed
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_COMPAT_PROFILE);

	// Open a window and create its OpenGL context
	window = glfwCreateWindow(1024, 768, "Rotation Estimation Demo", NULL, NULL);
	if (window == NULL) {
		fprintf(stderr, "Failed to open GLFW window. If you have an Intel GPU, they are not 3.3 compatible. Try the 2.1 version of the tutorials.\n");
		getchar();
		glfwTerminate();
		return -1;
	}
	glfwMakeContextCurrent(window);

	// Initialize GLEW
	glewExperimental = true; // Needed for core profile
	if (glewInit() != GLEW_OK) {
		fprintf(stderr, "Failed to initialize GLEW\n");
		getchar();
		glfwTerminate();
		return -1;
	}

	// Variables handled by the gui.
	int numSelectedNormals = 0;
	auto sourceRotation = glm::quat(1.0, 0.0, 0.0, 0.0);
	glm::quat estimatedRotation;
	glm::quat estimatedRotationAligned;
	float gaussianDeviationRadians = 0.0f;
	float error = 0.0f;

	// Setup gui.
	TwInit(TW_OPENGL_CORE, NULL);
	TwWindowSize(1024, 768);
	TwBar* tweakBar = TwNewBar("Demo");
	TwSetParam(tweakBar, NULL, "refresh", TW_PARAM_CSTRING, 1, "0.1");
	TwAddVarRW(tweakBar, "Randomize Normals", TW_TYPE_FLOAT, &gaussianDeviationRadians, " min=0.0 max=1.0 step=0.05 keyIncr='s' keyDecr='S' ");
	TwAddVarRW(tweakBar, "Num Selected Normals", TW_TYPE_INT32, &numSelectedNormals, NULL);
	TwAddVarRW(tweakBar, "Source Rotation", TW_TYPE_QUAT4F, &sourceRotation, NULL);
	TwAddVarRW(tweakBar, "Estimated Rotation", TW_TYPE_QUAT4F, &estimatedRotation, NULL);
	TwAddVarRW(tweakBar, "Aligned Estimated Rotation", TW_TYPE_QUAT4F, &estimatedRotationAligned, NULL);
	TwAddVarRW(tweakBar, "Error", TW_TYPE_FLOAT, &error, " min=0.0 max=1.0 ");

	// Forward events to gui.
	glfwSetMouseButtonCallback(window, (GLFWmousebuttonfun)TwEventMouseButtonGLFW);
	glfwSetCursorPosCallback(window, (GLFWcursorposfun)TwEventMousePosGLFW);
	glfwSetScrollCallback(window, (GLFWscrollfun)TwEventMouseWheelGLFW);
	glfwSetKeyCallback(window, (GLFWkeyfun)TwEventKeyGLFW);
	glfwSetCharCallback(window, (GLFWcharfun)TwEventCharGLFW);

	// Ensure we can capture the escape key being pressed below
	glfwSetInputMode(window, GLFW_STICKY_KEYS, GL_TRUE);
	glfwSetCursorPos(window, 1024 / 2, 768 / 2);
	glClearColor(0.0f, 0.0f, 0.4f, 0.0f);

	runTests();

	// Icosphere quantization constants.
	constexpr auto icosphereSubdivisionDepth = 5u;
	constexpr auto rotationEstimationSteps = 6;
	const auto numBins = getIcosphereFacesCountAtDepth(icosphereSubdivisionDepth);

	// Create and compile our GLSL program from the shaders
	GLuint programID = compileShadersProgram(vertexShader, fragmentShader);

	// Vertex buffer to draw normals.
	GLuint vertexArrayId;
	glGenVertexArrays(1, &vertexArrayId);
	glBindVertexArray(vertexArrayId);

	GLuint vertexbuffer;
	glGenBuffers(1, &vertexbuffer);
	glBindBuffer(GL_ARRAY_BUFFER, vertexbuffer);
	glBufferData(GL_ARRAY_BUFFER, sizeof(glm::vec3) * numBins * 2, NULL, GL_DYNAMIC_DRAW_ARB);

	// Our hardware features one CUDA capable device.
	// Had we more, we'd iterate over available devices.
	cudaDeviceProp prop;
	cudaGetDeviceProperties(&prop, 0);
	std::cout << "Multi Processor Count:" << prop.multiProcessorCount << std::endl;
	DispatchArgs dispatchArgs(prop.multiProcessorCount);

	// Share graphics buffer with cuda.
	cudaGraphicsResource* vertexbufferResource;
	cudaGraphicsGLRegisterBuffer(&vertexbufferResource, vertexbuffer, cudaGraphicsMapFlagsNone);

	glm::vec3* devNormalVertices;
	size_t size;

	// Get a handle for our "MVP" uniform
	auto transformId = glGetUniformLocation(programID, "ModelViewProjection");

	// We use a timer to limit frame rate.
	auto lastTime = glfwGetTime();

	// Simple camera controller.
	auto orbitControls = OrbitCamera();

	constexpr size_t normalsBatchSize = 1024 * 8;
	std::vector<glm::vec3> normals(normalsBatchSize);

	// Cuda device buffers.
	DeviceBuffer<glm::vec3> normalsBuffer(normalsBatchSize);
	DeviceBuffer<uint32_t> binsBuffer(numBins, true);
	DeviceBuffer<glm::vec3> reconstructedNormalsBuffer(numBins);
	DeviceBuffer<float> scoresInBuffer(numBins);
	DeviceBuffer<float> scoresOutBuffer(numBins);

	do {
		// Generate a fresh batch of normals for the frame and upload them to the device.
		sampleRandomizedNormals(normals, sourceRotation, gaussianDeviationRadians);
		normalsBuffer.copyFrom(normals);

		// Cuda. (Compute.)
		{
			binsBuffer.clearMemory();

			// Normals icosphere quantization.
			quantizeNormals(dispatchArgs, normalsBuffer, binsBuffer, icosphereSubdivisionDepth);

			// Select populated bins and reconstruct normals.
			numSelectedNormals = reconstructFilteredNormals(dispatchArgs, binsBuffer, reconstructedNormalsBuffer, icosphereSubdivisionDepth, 1u);

			// Calculate vertices for visualization based on quantization bins.
			cudaGraphicsMapResources(1, &vertexbufferResource, NULL);
			cudaGraphicsResourceGetMappedPointer((void**)&devNormalVertices, &size, vertexbufferResource);
			evaluateNormalsEndpoints(dispatchArgs, reconstructedNormalsBuffer, devNormalVertices, numSelectedNormals);
			cudaGraphicsUnmapResources(1, &vertexbufferResource, NULL);

			// Infer rotation from selected normals.
			estimatedRotation = evaluateRotationPlane(dispatchArgs, reconstructedNormalsBuffer, scoresInBuffer, scoresOutBuffer, numSelectedNormals, rotationEstimationSteps);
			estimatedRotation = evaluateRotationBasis(dispatchArgs, reconstructedNormalsBuffer, scoresInBuffer, scoresOutBuffer, numSelectedNormals, rotationEstimationSteps, estimatedRotation);

			// Evaluate error.
			auto estimatedRotation3x3 = glm::mat3_cast(estimatedRotation);
			auto sourceRotation3x3 = glm::mat3_cast(sourceRotation);
			auto changeOfBasis = getMinDistChangeOfBasis(estimatedRotation3x3, sourceRotation3x3);
			auto estimatedRotationAligned3x3 = estimatedRotation3x3 * changeOfBasis;
			estimatedRotationAligned = glm::quat_cast(estimatedRotationAligned3x3);
			error = getDistanceBetweenRotationMatrices(estimatedRotationAligned3x3, sourceRotation3x3);
		}
		// OpenGl. (Render.)
		{
			glClear(GL_COLOR_BUFFER_BIT);

			glUseProgram(programID);

			glm::mat4 transform = orbitControls.GetModelViewProjection();
			glUniformMatrix4fv(transformId, 1, GL_FALSE, &transform[0][0]);

			// 1rst attribute buffer : vertices
			glEnableVertexAttribArray(0);
			glBindBuffer(GL_ARRAY_BUFFER, vertexbuffer);
			glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, (void*)0);

			// Times 2, 2 vertices per normal.
			glDrawArrays(GL_LINES, 0, numSelectedNormals * 2);

			glDisableVertexAttribArray(0);
		}

		// Draw gui.
		TwDraw();

		// Swap buffers
		glfwSwapBuffers(window);
		glfwPollEvents();

		// Process input.
		orbitControls.Update();

		// Limit framerate.
		while (glfwGetTime() < lastTime + 1.0 / 60.0)
		{
			// Nothing, wait.
		}
		lastTime = glfwGetTime();

	} // Check if the ESC key was pressed or the window was closed
	while (glfwGetKey(window, GLFW_KEY_ESCAPE) != GLFW_PRESS &&
		glfwWindowShouldClose(window) == 0);

	// Cleanup VBO
	glDeleteBuffers(1, &vertexbuffer);
	glDeleteVertexArrays(1, &vertexArrayId);
	glDeleteProgram(programID);

	// Cleanup gui.
	TwTerminate();

	// Close OpenGL window and terminate GLFW
	glfwTerminate();

	return 0;
}

