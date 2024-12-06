#include "raster/rasterization.h"
#include "io/png.h"
#include "common.h"

#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>

#include <cstdio>
#include <iostream>
#include <random>

void GLAPIENTRY debugMessageCallback(GLenum source,
                                     GLenum type,
                                     GLuint id,
                                     GLenum severity,
                                     GLsizei length,
                                     const GLchar* message,
                                     const void* userParam)
{
    std::cerr << "OpenGL Debug Message:\n";
    std::cerr << "Source: ";
    switch (source) {
        case GL_DEBUG_SOURCE_API:             std::cerr << "API"; break;
        case GL_DEBUG_SOURCE_WINDOW_SYSTEM:   std::cerr << "Window System"; break;
        case GL_DEBUG_SOURCE_SHADER_COMPILER: std::cerr << "Shader Compiler"; break;
        case GL_DEBUG_SOURCE_THIRD_PARTY:     std::cerr << "Third Party"; break;
        case GL_DEBUG_SOURCE_APPLICATION:     std::cerr << "Application"; break;
        case GL_DEBUG_SOURCE_OTHER:           std::cerr << "Other"; break;
    }
    std::cerr << "\nType: ";
    switch (type) {
        case GL_DEBUG_TYPE_ERROR:               std::cerr << "Error"; break;
        case GL_DEBUG_TYPE_DEPRECATED_BEHAVIOR: std::cerr << "Deprecated Behavior"; break;
        case GL_DEBUG_TYPE_UNDEFINED_BEHAVIOR:  std::cerr << "Undefined Behavior"; break;
        case GL_DEBUG_TYPE_PORTABILITY:         std::cerr << "Portability"; break;
        case GL_DEBUG_TYPE_PERFORMANCE:         std::cerr << "Performance"; break;
        case GL_DEBUG_TYPE_MARKER:              std::cerr << "Marker"; break;
        case GL_DEBUG_TYPE_PUSH_GROUP:          std::cerr << "Push Group"; break;
        case GL_DEBUG_TYPE_POP_GROUP:           std::cerr << "Pop Group"; break;
        case GL_DEBUG_TYPE_OTHER:               std::cerr << "Other"; break;
    }
    std::cerr << "\nSeverity: ";
    switch (severity) {
        case GL_DEBUG_SEVERITY_HIGH:         std::cerr << "High"; break;
        case GL_DEBUG_SEVERITY_MEDIUM:       std::cerr << "Medium"; break;
        case GL_DEBUG_SEVERITY_LOW:          std::cerr << "Low"; break;
        case GL_DEBUG_SEVERITY_NOTIFICATION: std::cerr << "Notification"; break;
    }
    std::cerr << "\nMessage: " << message << std::endl;
    if (severity == GL_DEBUG_SEVERITY_HIGH || severity == GL_DEBUG_SEVERITY_MEDIUM) {
        assert (0);
    }
}

// TODO: https://developer.nvidia.com/nsight-perf-sdk/get-started

namespace GLState {
    static GLuint programid = -1;
    static GLFWwindow* window = nullptr;
    static GLuint fbo = -1;
    static GLuint texture = -1;
    static GLuint VertexArrayID = -1;
    static GLuint vertexbuffer = -1;
    static GLuint textureID = -1;
    static GLuint fragmentSSBO = -1, counterSSBO = -1;
};

struct FragData {
    glm::vec3 color;
    GLuint id;
};

float randf01() { // TODO: REFACTOR THIS
    static thread_local std::random_device rd;
    static thread_local std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(0.0, 1.0);
    return dis(gen);
}

static void gl_setup(int width, int height, int tex_width, int tex_height) {
    if (GLState::programid != -1) {
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
	GLState::window = glfwCreateWindow( 256, 256, "Rasterization context", NULL, NULL); // window will be hidden, resolution doesn't matter
	if( GLState::window == NULL ){
		fprintf( stderr, "Failed to open GLFW window. If you have an Intel GPU, they are not 3.3 compatible. Try the 2.1 version of the tutorials.\n" );
		getchar();
		glfwTerminate();
		return;
	}
	glfwMakeContextCurrent(GLState::window);

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

	// Shader source
    /* Note on viewport: the code considers (0, 0) the top-left origin and has y grow downwards.
     * This contradicts the typical arrangement of the OpenGL Viewport.
     * Instead of flipping the axis when drawing, we leave it unchanged because glReadPixels also
     * returns the opposite row-order that we expect, so both cancel out and correct the vertical flip. */
	std::string VertexShaderCode = R"(#version 430 core 
layout(location = 0) in vec2 vertexPosition;

out vec2 tex_coord;

void main() {
	gl_Position = vec4((vertexPosition.x - 0.5) * 2.0, (vertexPosition.y - 0.5) * 2.0, 0.0, 1.0);
    tex_coord = vertexPosition.xy;
}
)";

    std::string FragmentShaderCode = R"(#version 430 core
out vec4 out_colour;
in vec2 tex_coord;

uniform sampler2D tex;

struct FragData {
    vec3 color;
    uint id;
};

layout(std430, binding = 0) buffer FragmentBuffer {
    FragData fragments[];
};

layout(std430, binding = 1) buffer GlobalCounter {
    uint globalCounter;
};

void main() {
	vec4 texColor = texture(tex, tex_coord);

    uint index = atomicAdd(globalCounter, 1);

    fragments[index].color = texColor.xyz;
    fragments[index].id = gl_PrimitiveID + 1;

    out_colour = vec4(texColor.xyz, 0.25);
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

	GLState::programid = ProgramID;

    glViewport(0, 0, width, height);

    glGenFramebuffers(1, &GLState::fbo);
    glBindFramebuffer(GL_FRAMEBUFFER, GLState::fbo);

    glGenTextures(1, &GLState::texture);
    glBindTexture(GL_TEXTURE_2D, GLState::texture);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, tex_width, tex_height, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, GLState::texture, 0);

    glBindFramebuffer(GL_FRAMEBUFFER, GLState::fbo);
    glGenVertexArrays(1, &GLState::VertexArrayID);
    glBindVertexArray(GLState::VertexArrayID);

    glEnable(GL_BLEND);
    glBlendFuncSeparate(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA, GL_ONE, GL_ONE_MINUS_SRC_ALPHA);

    glGenBuffers(1, &GLState::vertexbuffer);
    glBindBuffer(GL_ARRAY_BUFFER, GLState::vertexbuffer);

    glGenTextures(1, &GLState::textureID);
    
    glGenBuffers(1, &GLState::fragmentSSBO);
    glGenBuffers(1, &GLState::counterSSBO);

    glUseProgram(GLState::programid);

    glEnable(GL_DEBUG_OUTPUT);
    glEnable(GL_DEBUG_OUTPUT_SYNCHRONOUS); // Make sure errors are outputted immediately

    // Set the debug message callback
    glDebugMessageCallback(debugMessageCallback, nullptr);
}

void gl_reset_ssbo(int width, int height, int num_tris) {
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, GLState::fragmentSSBO);
    int NFRAG = width * height * num_tris / 2;
    int NFRAGB = NFRAG * sizeof(FragData);
    glBufferData(GL_SHADER_STORAGE_BUFFER, NFRAGB, nullptr, GL_DYNAMIC_DRAW);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, GLState::fragmentSSBO);

    GLuint globalCounter = 0;
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, GLState::counterSSBO);
    glBufferData(GL_SHADER_STORAGE_BUFFER, sizeof(GLuint), &globalCounter, GL_DYNAMIC_DRAW);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, GLState::counterSSBO);
}

void gl_upload_triangles(const std::vector<geometry::triangle>& triangles) {
    glBindVertexArray(GLState::VertexArrayID);
    glBindBuffer(GL_ARRAY_BUFFER, GLState::vertexbuffer);
    std::vector<float> vertex_buffer_data;
    vertex_buffer_data.reserve(triangles.size());
    for (int i = 0; i < triangles.size(); i++) {
        auto tri = triangles[i];
        for (int j = 0; j < 3; j++) {
            vertex_buffer_data.push_back(tri[j].x);
            vertex_buffer_data.push_back(tri[j].y);
        }
    }
    // TODO: maybe can just send reinterpret_cast<tris.data()> ?

    glBufferData(GL_ARRAY_BUFFER, sizeof(float) * vertex_buffer_data.size(), vertex_buffer_data.data(), GL_STATIC_DRAW);

    glEnableVertexAttribArray(0);
    glBindBuffer(GL_ARRAY_BUFFER, GLState::vertexbuffer);
    glVertexAttribPointer(
        0,              
        2,             
        GL_FLOAT,       
        GL_FALSE,       
        2*sizeof(float),
        (void*)0
    );
}

void gl_upload_texture(const io::Image<io::RGB255>& tex) {
    glBindTexture(GL_TEXTURE_2D, GLState::textureID);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, tex.width(), tex.height(), 0, GL_RGB, GL_UNSIGNED_BYTE, tex.data());

    GLint texLoc = glGetUniformLocation(GLState::programid, "tex");

    glActiveTexture(GL_TEXTURE0 + 0);
    glBindTexture(GL_TEXTURE_2D, GLState::textureID);
    glUniform1i(texLoc, 0);
}

void gl_draw(int num_triangles) {
    glClearColor(1.f, 1.f, 1.f, 1.f);
    glClear( GL_COLOR_BUFFER_BIT );

    glDrawArrays(GL_TRIANGLES, 0, 3 * num_triangles);
}

void gl_get_pixels(io::Image<io::RGBA255> output) {
    glBindBuffer(GL_ARRAY_BUFFER, GLState::vertexbuffer);
    glBindFramebuffer(GL_FRAMEBUFFER, GLState::fbo);
    glDisableVertexAttribArray(0);

    glPixelStorei(GL_PACK_ROW_LENGTH, 0);
    glPixelStorei(GL_PACK_SKIP_PIXELS, 0);
    glPixelStorei(GL_PACK_SKIP_ROWS, 0);
    glPixelStorei(GL_PACK_ALIGNMENT, 1);
    glReadBuffer(GL_BACK_LEFT);

    glReadPixels(0, 0, output.width(), output.height(), GL_RGBA, GL_UNSIGNED_BYTE, (unsigned char*)output.data());
}

int main(int argc, char** argv) {
    int image_resolution = 128;

    io::Image<io::RGBA255> image(image_resolution, image_resolution, io::RGBA255{100, 0, 0, 255});

    auto background = io::load_png_rgb("background.png");
    auto target = io::load_png_rgb("lisa.png");
    gl_setup(image.width(), image.height(), target.width(), target.height());
    for (int trial = 0; trial < 1; trial++) {
        int num_tris = 10000;
        std::vector<geometry::triangle> tris;
        tris.reserve(num_tris);
        for (int i = 0; i < num_tris; i++) {
            geometry::triangle tri{{randf01(), randf01()}, {randf01(), randf01()}, {randf01(), randf01()}};
            tris.push_back(tri);
        }

        gl_reset_ssbo(image.width(), image.height(), num_tris);
        gl_upload_triangles(tris);
        gl_upload_texture(target);
        gl_draw(num_tris);
        gl_get_pixels(image);
    }

    io::save_png_rgba("output.png", image);

    return 0;
}

