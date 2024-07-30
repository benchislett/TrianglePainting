#include "raster/rasterization.h"

#include <GL/glew.h>
#include <GLFW/glfw3.h>

static GLuint programid = -1;
static GLFWwindow* window = nullptr;

static void init() {
    if (programid != -1) {
        return;
    }

    // Set up the GLFW context

    if( !glfwInit() )
	{
		fprintf( stderr, "Failed to initialize GLFW\n" );
		getchar();
        return;
	}

	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
	glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE); // To make macOS happy; should not be needed
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
	glfwWindowHint(GLFW_VISIBLE, GLFW_FALSE); // window will be hidden

	// Open a window and create its OpenGL context
	window = glfwCreateWindow( 256, 256, "Rasterization context", NULL, NULL); // window will be hidden, resolution doesn't matter
	if( window == NULL ){
		fprintf( stderr, "Failed to open GLFW window. If you have an Intel GPU, they are not 3.3 compatible. Try the 2.1 version of the tutorials.\n" );
		getchar();
		glfwTerminate();
		return;
	}
	glfwMakeContextCurrent(window);

    // Initialize GLEW
    glewExperimental = true; // Needed for core profile
	if (glewInit() != GLEW_OK) {
		fprintf(stderr, "Failed to initialize GLEW\n");
		getchar();
		glfwTerminate();
		return;
	}

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

	programid = ProgramID;
}

namespace raster {

    void rasterize_triangles_rgba_2d_opengl(const std::vector<geometry2d::triangle>& triangles, const std::vector<RGBA255>& colours, ImageBuffer<RGBA255>& image) {
        init();

        glViewport(0, 0, image.width, image.height);

        GLuint fbo;
        glGenFramebuffers(1, &fbo);
        glBindFramebuffer(GL_FRAMEBUFFER, fbo);

        // set up a framebuffer for off-screen rendering
        GLuint texture;
        glGenTextures(1, &texture);
        glBindTexture(GL_TEXTURE_2D, texture);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, image.width, image.height, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, texture, 0);	

        glBindFramebuffer(GL_FRAMEBUFFER, fbo);

        GLuint VertexArrayID;
        glGenVertexArrays(1, &VertexArrayID);
        glBindVertexArray(VertexArrayID);

        // enable blending
        glEnable(GL_BLEND);
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

        std::vector<float> vertex_buffer_data;
        for (int i = 0; i < triangles.size(); i++) {
            auto tri = triangles[i];
            for (int j = 0; j < 3; j++) {
                vertex_buffer_data.push_back(tri[j].x);
                vertex_buffer_data.push_back(tri[j].y);
                vertex_buffer_data.push_back(colours[i].r / 255.0f);
                vertex_buffer_data.push_back(colours[i].g / 255.0f);
                vertex_buffer_data.push_back(colours[i].b / 255.0f);
                vertex_buffer_data.push_back(colours[i].a / 255.0f);
            }
        }

        GLuint vertexbuffer;
        glGenBuffers(1, &vertexbuffer);
        glBindBuffer(GL_ARRAY_BUFFER, vertexbuffer);
        glBufferData(GL_ARRAY_BUFFER, sizeof(float) * vertex_buffer_data.size(), vertex_buffer_data.data(), GL_STATIC_DRAW);

        glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
        glClear( GL_COLOR_BUFFER_BIT );

        // Use the shader
        glUseProgram(programid);

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
        glDrawArrays(GL_TRIANGLES, 0, 3 * triangles.size());

        glDisableVertexAttribArray(0);

        glPixelStorei(GL_PACK_ROW_LENGTH, 0);
        glPixelStorei(GL_PACK_SKIP_PIXELS, 0);
        glPixelStorei(GL_PACK_SKIP_ROWS, 0);
        glPixelStorei(GL_PACK_ALIGNMENT, 1);
        glReadBuffer(GL_BACK_LEFT);

        glReadPixels(0, 0, image.width, image.height, GL_RGBA, GL_UNSIGNED_BYTE, (unsigned char*)image.data.data());

        // mirror the image vertically
        for (int y = 0; y < image.height / 2; y++) {
            for (int x = 0; x < image.width; x++) {
                std::swap(image.data[y * image.width + x], image.data[(image.height - 1 - y) * image.width + x]);
            }
        }

        // Cleanup VBO
        glDeleteBuffers(1, &vertexbuffer);
        glDeleteVertexArrays(1, &VertexArrayID);
    }
}