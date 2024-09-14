#include "raster/rasterization.h"
#include "io/png.h"

#include <GL/glew.h>
#include <GLFW/glfw3.h>

#include <cstdio>
#include <iostream>

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

	// Shader source
    /* Note on viewport: the code considers (0, 0) the top-left origin and has y grow downwards.
     * This contradicts the typical arrangement of the OpenGL Viewport.
     * Instead of flipping the axis when drawing, we leave it unchanged because glReadPixels also
     * returns the opposite row-order that we expect, so both cancel out and correct the vertical flip. */
	std::string VertexShaderCode = R"(#version 330 core 
layout(location = 0) in vec2 vertexPosition;

out vec4 colour;
out vec2 tex_coord;

void main() {
	gl_Position = vec4((vertexPosition.x - 0.5) * 2.0, (vertexPosition.y - 0.5) * 2.0, 0.0, 1.0);
    colour = vec4(1, 0, 0, 1);
    tex_coord = vertexPosition.xy;
}
)";

    std::string FragmentShaderCode = R"(#version 330 core
in vec4 colour;
out vec4 out_colour;
in vec2 tex_coord;

uniform sampler2D tex;

void main() {
	out_colour = vec4(texture2D(tex, tex_coord).xyz, 1.f);
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

void test_rast(const std::vector<geometry2d::triangle>& triangles, io::Image<io::RGBA255>& image, const io::Image<io::RGB255>& background, const io::Image<io::RGB255>& target) {
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

    /* Alpha Blending
        * Blending the source and destination colours based on the source alpha value
        * As is standard for "over" compositing with non-premultiplied alpha
        * And determining the new alpha value according to the usual case
        * See https://en.wikipedia.org/wiki/Alpha_compositing
        * And https://registry.khronos.org/OpenGL-Refpages/gl4/html/glBlendFuncSeparate.xhtml
        */
    glEnable(GL_BLEND);
    glBlendFuncSeparate(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA, GL_ONE, GL_ONE_MINUS_SRC_ALPHA);

    glClearColor(1.f, 1.f, 1.f, 1.f);
    glClear( GL_COLOR_BUFFER_BIT );

    std::vector<float> vertex_buffer_data;
    for (int i = 0; i < triangles.size(); i++) {
        auto tri = triangles[i];
        for (int j = 0; j < 3; j++) {
            vertex_buffer_data.push_back(tri[j].x);
            vertex_buffer_data.push_back(tri[j].y);
        }
    }

    GLuint vertexbuffer;
    glGenBuffers(1, &vertexbuffer);
    glBindBuffer(GL_ARRAY_BUFFER, vertexbuffer);
    glBufferData(GL_ARRAY_BUFFER, sizeof(float) * vertex_buffer_data.size(), vertex_buffer_data.data(), GL_STATIC_DRAW);

    GLuint textureID[1];
    glGenTextures(1, &textureID[0]);

    // Use the shader
    glUseProgram(programid);

    glBindTexture(GL_TEXTURE_2D, textureID[0]);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, target.width, target.height, 0, GL_RGB, GL_UNSIGNED_BYTE, target.data.data());

    GLint texLoc = glGetUniformLocation(programid, "tex");

    glActiveTexture(GL_TEXTURE0 + 0);
    glBindTexture(GL_TEXTURE_2D, textureID[0]);
    glUniform1i(texLoc, 0);

    // Vertex position buffer
    glEnableVertexAttribArray(0);
    glBindBuffer(GL_ARRAY_BUFFER, vertexbuffer);
    glVertexAttribPointer(
        0,              
        2,             
        GL_FLOAT,       
        GL_FALSE,       
        2*sizeof(float),
        (void*)0
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

    // Cleanup VBO
    glDeleteBuffers(1, &vertexbuffer);
    glDeleteVertexArrays(1, &VertexArrayID);
}

int main(int argc, char** argv) {
    int image_resolution = 1024;

    io::Image<io::RGBA255> image;
    image.width = image_resolution;
    image.height = image_resolution;

    image.data.resize(image.width * image.height);
    std::fill(image.data.begin(), image.data.end(), io::RGBA255{0, 0, 0, 255});

    geometry2d::triangle tri{{0.1, 0.1}, {0.5, 0.1}, {0.5, 0.5}};
    std::vector<geometry2d::triangle> tris = {tri};

    test_rast(tris, image, io::load_png_rgb("background.png"), io::load_png_rgb("lisa.png"));

    io::save_png_rgba("output.png", image);

    return 0;
}

