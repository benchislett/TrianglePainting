#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <iostream>
#include <vector>
#include <random>
#include <chrono>

#include "io/png.h"
#include "geometry/types.h"

#define WIN_X 200
#define WIN_Y WIN_X

float randf(float min, float max) {
    static thread_local std::random_device rd;
    static thread_local std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(min, max);
    return dis(gen);
}

// Global start time
static std::chrono::time_point<std::chrono::high_resolution_clock> _perf_start;

struct PerfInstrumenter {
    static void register_start() {
        _perf_start = std::chrono::high_resolution_clock::now();
    }

    static void register_step(const std::string& step_name) {
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end - _perf_start;
        std::cout << step_name << " took " << elapsed.count() << "s" << std::endl;
        _perf_start = std::chrono::high_resolution_clock::now();
    }
};

// Vertex and fragment shaders
const char* vertexShaderSource = R"(
#version 430 core
layout(location = 0) in vec2 position;
out vec2 fragTexCoord;

void main() {
    gl_Position = vec4(position.x, -position.y, 0.0, 1.0);

    // Normalize texture coordinates
    fragTexCoord = (position + 1.0) / 2.0;
    fragTexCoord.y = 1.0 - fragTexCoord.y;
}
)";

const char* fragmentShaderSource = R"(
#version 430 core
out vec4 fragColor;

uniform sampler2D myTexture;
in vec2 fragTexCoord;

struct FragmentData {
    vec4 color;
    uint triangleID;
};

layout(std430, binding = 0) buffer FragmentBuffer {
    FragmentData fragments[];
};

layout(std430, binding = 1) buffer GlobalCounter {
    uint globalCounter;
};

void main() {
    vec4 texColor = texture(myTexture, fragTexCoord);
    
    // Use atomicAdd to get the next available position in the buffer
    uint index = atomicAdd(globalCounter, 1);
    
    // Store fragment color and triangle ID
    fragments[index].color = texColor;
    fragments[index].triangleID = gl_PrimitiveID + 1;

    // Output fragment color to the screen (optional)
    fragColor = texColor;
}
)";

// Compute shader to accumulate and calculate the average color
const char* computeShaderSource = R"(
#version 430 core
layout(local_size_x = 256) in;

struct FragmentData {
    vec4 color;
    uint triangleID;
};

layout(std430, binding = 0) buffer FragmentBuffer {
    FragmentData fragments[];
};

layout(std430, binding = 1) buffer GlobalCounter {
    uint globalCounter;
};

layout(std430, binding = 2) buffer TriangleResultBuffer {
    vec4 averageColors[];
};

shared uint accumR;
shared uint accumG;
shared uint accumB;
shared uint fragmentCount;

void main() {
    uint triangleID = gl_WorkGroupID.x;
    uint localID = gl_LocalInvocationID.x;

    if (localID == 0) {
        accumR = 0;
        accumG = 0;
        accumB = 0;
        fragmentCount = 0;
    }
    barrier();

    uint totalFragments = globalCounter;

    // Loop through all fragments and accumulate for the current triangle
    for (uint i = localID; i < totalFragments; i += gl_WorkGroupSize.x) {
        if (fragments[i].triangleID - 1 == triangleID) {
            atomicAdd(fragmentCount, 1);
            // atomicAdd(colorSum, fragments[i].color);
            atomicAdd(accumR, uint(fragments[i].color.x * 255));
            atomicAdd(accumG, uint(fragments[i].color.y * 255));
            atomicAdd(accumB, uint(fragments[i].color.z * 255));
        }
    }
    barrier();

    // Compute and store the average color for this triangle
    if (localID == 0 && fragmentCount > 0) {
        averageColors[triangleID] = vec4(accumR / float(fragmentCount), accumG / float(fragmentCount), accumB / float(fragmentCount), 1.f);
    }
}
)";

// Utility to compile and link shaders
GLuint CompileShader(const char* source, GLenum shaderType) {
    GLuint shader = glCreateShader(shaderType);
    glShaderSource(shader, 1, &source, nullptr);
    glCompileShader(shader);

    GLint success;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
    if (!success) {
        char infoLog[512];
        glGetShaderInfoLog(shader, 512, nullptr, infoLog);
        std::cerr << "Shader Compilation Error: " << infoLog << std::endl;
    }

    return shader;
}

// Utility to check linking status
void CheckLinking(GLuint program) {
    GLint success;
    glGetProgramiv(program, GL_LINK_STATUS, &success);
    if (!success) {
        char infoLog[512];
        glGetProgramInfoLog(program, 512, nullptr, infoLog);
        std::cerr << "Program Linking Error: " << infoLog << std::endl;
    }
}

void PrintResults(const std::vector<glm::vec4>& averageColors, int head = -1) {
    std::cout << "Triangle Average Colors:" << std::endl;
    int num_to_print = head == -1 ? averageColors.size() : head;
    for (size_t i = 0; i < num_to_print; ++i) {
        std::cout << "Triangle " << i << ": (" 
                  << averageColors[i].r << ", "
                  << averageColors[i].g << ", "
                  << averageColors[i].b << ", "
                  << averageColors[i].a << ")" << std::endl;
    }
}

struct FragmentData {
    glm::vec4 color;
    uint triangleID;
    uint pad[3];
};

void PrintFragment(const FragmentData& fragment) {
    printf("%d | %f %f %f %f\n", fragment.triangleID, fragment.color.x, fragment.color.y, fragment.color.z, fragment.color.w);
}

int main() {
    PerfInstrumenter::register_start();

    // Initialize GLFW and GLEW
    if (!glfwInit()) {
        std::cerr << "Failed to initialize GLFW" << std::endl;
        return -1;
    }
    
    GLFWwindow* window = glfwCreateWindow(WIN_X, WIN_Y, "OpenGL Fragment Accumulation", nullptr, nullptr);
    if (!window) {
        std::cerr << "Failed to create GLFW window" << std::endl;
        glfwTerminate();
        return -1;
    }
    glfwMakeContextCurrent(window);
    
    glewExperimental = GL_TRUE;
    if (glewInit() != GLEW_OK) {
        std::cerr << "Failed to initialize GLEW" << std::endl;
        return -1;
    }

    PerfInstrumenter::register_step("GL Init");

    int num_tris = 5000;
    std::vector<geometry::triangle> tris;

    for (int i = 0; i < num_tris; i++) {
        tris.emplace_back(geometry::triangle{
            {randf(-1, 1), randf(-1, 1)},
            {randf(-1, 1), randf(-1, 1)},
            {randf(-1, 1), randf(-1, 1)}
        });
    }

    // Setup VBO and VAO
    GLuint VBO, VAO;
    glGenVertexArrays(1, &VAO);
    glGenBuffers(1, &VBO);
    
    glBindVertexArray(VAO);
    
    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(geometry::triangle) * tris.size(), (const float*)tris.data(), GL_STATIC_DRAW);

    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 2 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);
    
    glBindVertexArray(0);

    // Create shaders
    GLuint vertexShader = CompileShader(vertexShaderSource, GL_VERTEX_SHADER);
    GLuint fragmentShader = CompileShader(fragmentShaderSource, GL_FRAGMENT_SHADER);
    
    GLuint shaderProgram = glCreateProgram();
    glAttachShader(shaderProgram, vertexShader);
    glAttachShader(shaderProgram, fragmentShader);
    glLinkProgram(shaderProgram);
    CheckLinking(shaderProgram);

    // Create the compute shader
    GLuint computeShader = CompileShader(computeShaderSource, GL_COMPUTE_SHADER);
    GLuint computeProgram = glCreateProgram();
    glAttachShader(computeProgram, computeShader);
    glLinkProgram(computeProgram);
    CheckLinking(computeProgram);
    
    // Cleanup shaders after linking
    glDeleteShader(vertexShader);
    glDeleteShader(fragmentShader);
    glDeleteShader(computeShader);
    
    // Load texture data (assume `unsigned char* textureData` points to RGB data)
    io::Image<io::RGB255> textureIm = io::load_png_rgb("lisa.png");
    GLuint texture;
    glGenTextures(1, &texture);
    glBindTexture(GL_TEXTURE_2D, texture);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, textureIm.width(), textureIm.height(), 0, GL_RGB, GL_UNSIGNED_BYTE, textureIm.data());
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

    // Setup fragment buffer and global counter
    GLuint fragmentSSBO, counterSSBO;
    glGenBuffers(1, &fragmentSSBO);
    glGenBuffers(1, &counterSSBO);
    
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, fragmentSSBO);
    glBufferData(GL_SHADER_STORAGE_BUFFER, WIN_X * WIN_Y * tris.size() * sizeof(FragmentData), nullptr, GL_DYNAMIC_DRAW);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, fragmentSSBO);

    GLuint globalCounter = 0;
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, counterSSBO);
    glBufferData(GL_SHADER_STORAGE_BUFFER, sizeof(GLuint), &globalCounter, GL_DYNAMIC_DRAW);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, counterSSBO);

    // Set up output buffer for compute shader
    GLuint resultSSBO;
    glGenBuffers(1, &resultSSBO);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, resultSSBO);
    glBufferData(GL_SHADER_STORAGE_BUFFER, tris.size() * sizeof(glm::vec4), nullptr, GL_DYNAMIC_DRAW);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, resultSSBO);

    PerfInstrumenter::register_step("GL Setup");

    // Main rendering loop
    while (!glfwWindowShouldClose(window)) {
        std::vector<float> clearData(tris.size() * 4, 0);
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, resultSSBO);
        glBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, clearData.size() * sizeof(float), clearData.data());

        glClearColor(0.8f, 0.8f, 0.8f, 1.0f );
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        PerfInstrumenter::register_step("GL Clear");

        // Render pass
        glUseProgram(shaderProgram);
        glBindVertexArray(VAO);
        glDrawArrays(GL_TRIANGLES, 0, tris.size() * 3);

        PerfInstrumenter::register_step("GL Draw");

        /*
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, fragmentSSBO);
        GLint size_bytes;
        glGetBufferParameteriv(GL_SHADER_STORAGE_BUFFER, GL_BUFFER_SIZE, &size_bytes);
        std::cout << size_bytes << std::endl;
        std::vector<FragmentData> fragmentData(size_bytes / sizeof(FragmentData));
        glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, size_bytes, fragmentData.data());
        std::cout << fragmentData.size() << std::endl;

        // for (auto x : fragmentData) {
        //     PrintFragment(x);
        // }
        // std::cout << '\n';
        
        PerfInstrumenter::register_step("GL Fetch Fragments");
        */

        
        // Compute pass
        glUseProgram(computeProgram);
        glDispatchCompute(tris.size(), 1, 1);
        glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

        PerfInstrumenter::register_step("GL Compute");
        
        // Retrieve and print results
        std::vector<glm::vec4> averageColors(tris.size());
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, resultSSBO);
        glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, tris.size() * sizeof(glm::vec4), averageColors.data());

        PerfInstrumenter::register_step("GL Fetch Results");
        
        PrintResults(averageColors, 5);

        // Poll for events and swap buffers
        glfwSwapBuffers(window);
        glfwPollEvents();

        io::Image<io::RGBA255> output(WIN_X, WIN_Y);
        glBindBuffer(GL_ARRAY_BUFFER, VBO);
        glDisableVertexAttribArray(0);

        glPixelStorei(GL_PACK_ROW_LENGTH, 0);
        glPixelStorei(GL_PACK_SKIP_PIXELS, 0);
        glPixelStorei(GL_PACK_SKIP_ROWS, 0);
        glPixelStorei(GL_PACK_ALIGNMENT, 1);
        glReadBuffer(GL_BACK_LEFT);

        glReadPixels(0, 0, output.width(), output.height(), GL_RGBA, GL_UNSIGNED_BYTE, (unsigned char*)output.data());

        PerfInstrumenter::register_step("GL Read Pixels");

        io::save_png_rgba("output.png", output);

        PerfInstrumenter::register_step("Write Image");

        break;
    }

    // Cleanup
    glDeleteVertexArrays(1, &VAO);
    glDeleteBuffers(1, &VBO);
    glDeleteProgram(shaderProgram);
    glDeleteProgram(computeProgram);
    glDeleteBuffers(1, &fragmentSSBO);
    glDeleteBuffers(1, &counterSSBO);
    glDeleteBuffers(1, &resultSSBO);

    glfwTerminate();

    PerfInstrumenter::register_step("GL Cleanup");

    return 0;
}
