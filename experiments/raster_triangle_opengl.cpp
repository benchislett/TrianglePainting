// Include standard headers
#include <string>
#include <vector>
#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <fstream>

#include "io/image.h"

#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <nlohmann/json.hpp>

GLuint compile_shaders() {
	// Create the shaders
	GLuint VertexShaderID = glCreateShader(GL_VERTEX_SHADER);
	GLuint FragmentShaderID = glCreateShader(GL_FRAGMENT_SHADER);

	// Source code
	std::string VertexShaderCode = R"(#version 330 core 
layout(location = 0) in vec2 vertexPosition;
layout(location = 1) in vec4 rgba;

out vec4 colour;

void main() {
	gl_Position = vec4((vertexPosition.x - 0.5) * 2.0, (vertexPosition.y - 0.5) * -2.0, 0.0, 1.0);
    colour = rgba;
}
)";

    std::string FragmentShaderCode = R"(#version 330 core
in vec4 colour;
out vec4 out_colour;

void main() {
	out_colour = colour;
}
)";

	GLint Result = GL_FALSE;
	int InfoLogLength;

	char const * VertexSourcePointer = VertexShaderCode.c_str();
	glShaderSource(VertexShaderID, 1, &VertexSourcePointer , NULL);
	glCompileShader(VertexShaderID);

	// Check Vertex Shader
	glGetShaderiv(VertexShaderID, GL_COMPILE_STATUS, &Result);
	glGetShaderiv(VertexShaderID, GL_INFO_LOG_LENGTH, &InfoLogLength);
	if ( InfoLogLength > 0 ){
		std::vector<char> VertexShaderErrorMessage(InfoLogLength+1);
		glGetShaderInfoLog(VertexShaderID, InfoLogLength, NULL, &VertexShaderErrorMessage[0]);
		fprintf(stderr, "%s\n", &VertexShaderErrorMessage[0]);
	}

	// Compile Fragment Shader
	char const * FragmentSourcePointer = FragmentShaderCode.c_str();
	glShaderSource(FragmentShaderID, 1, &FragmentSourcePointer , NULL);
	glCompileShader(FragmentShaderID);

	// Check Fragment Shader
	glGetShaderiv(FragmentShaderID, GL_COMPILE_STATUS, &Result);
	glGetShaderiv(FragmentShaderID, GL_INFO_LOG_LENGTH, &InfoLogLength);
	if ( InfoLogLength > 0 ){
		std::vector<char> FragmentShaderErrorMessage(InfoLogLength+1);
		glGetShaderInfoLog(FragmentShaderID, InfoLogLength, NULL, &FragmentShaderErrorMessage[0]);
		fprintf(stderr, "%s\n", &FragmentShaderErrorMessage[0]);
	}

	// Link the program
	GLuint ProgramID = glCreateProgram();
	glAttachShader(ProgramID, VertexShaderID);
	glAttachShader(ProgramID, FragmentShaderID);
	glLinkProgram(ProgramID);

	// Check the program
	glGetProgramiv(ProgramID, GL_LINK_STATUS, &Result);
	glGetProgramiv(ProgramID, GL_INFO_LOG_LENGTH, &InfoLogLength);
	if ( InfoLogLength > 0 ){
		std::vector<char> ProgramErrorMessage(InfoLogLength+1);
		glGetProgramInfoLog(ProgramID, InfoLogLength, NULL, &ProgramErrorMessage[0]);
		fprintf(stderr, "%s\n", &ProgramErrorMessage[0]);
	}

	// Clean up
	glDetachShader(ProgramID, VertexShaderID);
	glDetachShader(ProgramID, FragmentShaderID);
	
	glDeleteShader(VertexShaderID);
	glDeleteShader(FragmentShaderID);

	return ProgramID;
}

int main(int argc, char** argv)
{
	/* Config options */
	int image_resolution = 512;
	std::vector<float> vertex_buffer_data;
	float background[4] = {1.0f, 1.0f, 1.0f, 0.0f};
	if (argc > 1) {
		std::string input_filename = argv[1];
		auto json = nlohmann::json::parse(std::ifstream(input_filename));
		for (auto& tri : json["triangles"]) {
			for (int i = 0; i < 3; i++) {
				vertex_buffer_data.push_back(tri["vertices"][i][0]);
				vertex_buffer_data.push_back(tri["vertices"][i][1]);
				vertex_buffer_data.push_back(tri["colour"][0]);
				vertex_buffer_data.push_back(tri["colour"][1]);
				vertex_buffer_data.push_back(tri["colour"][2]);
				vertex_buffer_data.push_back(tri["colour"][3]);
			}
		}
	} else {
		vertex_buffer_data = {
			0.25f, 0.25f, 1.0f, 0.f, 0.f, 1.f,
			0.75f, 0.25f, 0.0f, 1.f, 0.f, 1.f,
			0.5f,  0.75f, 0.0f, 0.f, 1.f, 1.f,
			0.1f, 0.2f, 1.0f, 0.f, 0.f, 1.f,
			0.15f,  0.3f, 0.0f, 1.f, 0.f, 1.f,
			0.3f, 0.15f, 0.0f, 0.f, 1.f, 1.f,
		};
	}
	int num_triangles = vertex_buffer_data.size() / 6 / 3;

	if (argc > 2) {
		image_resolution = std::atoi(argv[2]);
	}
	std::string output_filename = "raster_triangle_opengl.png";
	if (argc > 3) {
		output_filename = argv[3];
	}

	// Initialize GLFW
	if( !glfwInit() )
	{
		fprintf( stderr, "Failed to initialize GLFW\n" );
		getchar();
		return -1;
	}

	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
	glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE); // To make macOS happy; should not be needed
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
	glfwWindowHint(GLFW_VISIBLE, GLFW_FALSE); // window will be hidden

	// Open a window and create its OpenGL context
	GLFWwindow* window = glfwCreateWindow( 256, 256, "Tutorial 02 - Red triangle", NULL, NULL); // window will be hidden, resolution doesn't matter
	if( window == NULL ){
		fprintf( stderr, "Failed to open GLFW window. If you have an Intel GPU, they are not 3.3 compatible. Try the 2.1 version of the tutorials.\n" );
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

	// Ensure we can capture the escape key being pressed below
	// glfwSetInputMode(window, GLFW_STICKY_KEYS, GL_TRUE);

	glViewport(0, 0, image_resolution, image_resolution);

	GLuint fbo;
	glGenFramebuffers(1, &fbo);
	glBindFramebuffer(GL_FRAMEBUFFER, fbo);

	// set up a framebuffer for off-screen rendering
	GLuint texture;
	glGenTextures(1, &texture);
	glBindTexture(GL_TEXTURE_2D, texture);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, image_resolution, image_resolution, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, texture, 0);	

	glBindFramebuffer(GL_FRAMEBUFFER, fbo);

	GLuint VertexArrayID;
	glGenVertexArrays(1, &VertexArrayID);
	glBindVertexArray(VertexArrayID);

	// Create and compile our GLSL program from the shaders
	GLuint programID = compile_shaders();

	// enable blending
	glEnable(GL_BLEND);
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

	GLuint vertexbuffer;
	glGenBuffers(1, &vertexbuffer);
	glBindBuffer(GL_ARRAY_BUFFER, vertexbuffer);
	glBufferData(GL_ARRAY_BUFFER, sizeof(float) * vertex_buffer_data.size(), vertex_buffer_data.data(), GL_STATIC_DRAW);

	glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
	glClear( GL_COLOR_BUFFER_BIT );

	// Use the shader
	glUseProgram(programID);

	// Vertex position buffer
	glEnableVertexAttribArray(0);
	glBindBuffer(GL_ARRAY_BUFFER, vertexbuffer);
	glVertexAttribPointer(
		0,              
		2,             
		GL_FLOAT,       
		GL_FALSE,       
		6*sizeof(float),
		(void*)0        
	);

	// Vertex colour buffer
	glEnableVertexAttribArray(1);
	glVertexAttribPointer(
		1,                  
		4,                  
		GL_FLOAT,           
		GL_FALSE,           
		6*sizeof(float),    
		(void*)(2*sizeof(float))
	);

	// Draw the triangles
	glDrawArrays(GL_TRIANGLES, 0, 3 * num_triangles);

	glDisableVertexAttribArray(0);

	glPixelStorei(GL_PACK_ROW_LENGTH, 0);
	glPixelStorei(GL_PACK_SKIP_PIXELS, 0);
	glPixelStorei(GL_PACK_SKIP_ROWS, 0);
    glPixelStorei(GL_PACK_ALIGNMENT, 1);
	glReadBuffer(GL_BACK_LEFT);
    // save a screenshot
    int width = image_resolution, height = image_resolution;
    // glfwGetFramebufferSize(window, &width, &height);
    std::vector<unsigned char> image(width * height * 4);
    glReadPixels(0, 0, width, height, GL_RGBA, GL_UNSIGNED_BYTE, image.data());
	// mirror the image vertically
	for (int y = 0; y < height / 2; y++) {
		for (int x = 0; x < width; x++) {
			for (int c = 0; c < 4; c++) {
				std::swap(image[(y * width + x) * 4 + c], image[((height - 1 - y) * width + x) * 4 + c]);
			}
		}
	}

    io::save_png(output_filename, {std::move(image), (unsigned int)width, (unsigned int)height, 4});

	// Cleanup VBO
	glDeleteBuffers(1, &vertexbuffer);
	glDeleteVertexArrays(1, &VertexArrayID);
	glDeleteProgram(programID);

	// Close OpenGL window and terminate GLFW
	glfwTerminate();

	return 0;
}
