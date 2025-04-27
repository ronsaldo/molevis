#ifndef MOLEVIS_HPP
#define MOLEVIS_HPP

#pragma once

#include "SDL.h"
#include "SDL_syswm.h"
#include "AGPU/agpu.hpp"
#include "AABox.hpp"
#include "DAABox.hpp"
#include "PeriodicTable.hpp"
#include "LennardJonesTable.hpp"
#include "AtomBondDescription.hpp"
#include "AtomDescription.hpp"
#include "AtomState.hpp"
#include "CameraState.hpp"
#include "ModelState.hpp"
#include "Vector2.hpp"
#include "Vector3.hpp"
#include "Vector4.hpp"
#include "Matrix4x4.hpp"
#include "Frustum.hpp"
#include <chemfiles.hpp>
#include <stdint.h>
#include <stdio.h>
#include <memory>
#include <vector>
#include <string>
#include <random>
#include <time.h>
#include <atomic>
#include <mutex>
#include <condition_variable>
#include <thread>

#ifdef _WIN32
inline int64_t getMicroseconds()
{
    // TODO: Use performance counters
    return SDL_GetTicks64()*1000;
}
#else
inline int64_t getMicroseconds()
{
    timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return int64_t(ts.tv_sec*1000000) + int64_t(ts.tv_nsec/1000);
}

#endif

// Units simulationTimeStep
const double SimulationTimeStep = 1e-3f; // Picoseconds
const double BoltzmannConstantSI = 1.380649e-23; // m^2.K^-1
const double TargetTemperature = 10; // Kelvin

inline double
lennardJonesPotential(double r, double sigma, double epsilon)
{
    return 4*epsilon*(pow(sigma/r, 12) - pow(sigma/r, 6));
}

inline double
lennardJonesDerivative(double r, double sigma, double epsilon)
{
    return 24*epsilon*(pow(sigma, 6)/pow(r, 7) - 2.0*pow(sigma, 12)/pow(r, 13));
}

inline double
morsePotential(double r, double De, double a, double re)
{
    double interior = (1 - exp(-a*(r - re)));
    return De*interior*interior;
}

inline double
morsePotentialDerivative(double r, double De, double a, double re)
{
    double innerExp = exp(-a*(r - re));
    return -2.0*a*De*(1.0 - innerExp)*innerExp;
}

inline double
hookPotential(double distance, double equilibriumDistance, double k)
{
    double delta = distance - equilibriumDistance;
    return 0.5*k * (delta*delta);
}

inline double
hookPotentialDerivative(double distance, double equilibriumDistance, double k)
{
    double delta = distance - equilibriumDistance;
    return k * delta;
}


struct UIElementQuad
{
    Vector2 position;
    Vector2 size;

    Vector4 color;

    uint32_t isGlyph;

    Vector2 fontPosition;
    Vector2 fontSize;
};

struct Random
{
    Random(int seed = 45)
    {
        rand.seed(seed);
    }

    float randFloat(float min, float max)
    {
        return float(std::uniform_real_distribution<> (min, max)(rand));
    }

    double randDouble(float min, float max)
    {
        return std::uniform_real_distribution<> (min, max)(rand);
    }

    Vector3 randVector3(const Vector3 &min, const Vector3 &max)
    {
        return Vector3{randFloat(min.x, max.x), randFloat(min.y, max.y), randFloat(min.z, max.z)};
    }

    DVector3 randDVector3(const Vector3 &min, const Vector3 &max)
    {
        return DVector3{randDouble(min.x, max.x), randDouble(min.y, max.y), randDouble(min.z, max.z)};
    }

    Vector4 randVector4(const Vector4 &min, const Vector4 &max)
    {
        return Vector4{randFloat(min.x, max.x), randFloat(min.y, max.y), randFloat(min.z, max.z), randFloat(min.w, max.w)};
    }

    uint32_t randUInt(uint32_t max)
    {
        return rand() % max;
    }

    std::mt19937 rand;
};

struct TrackedDeviceModel
{
    agpu_buffer_ref vertexBuffer;
    size_t vertexCount;
    agpu_vertex_binding_ref vertexBinding;

    agpu_buffer_ref indexBuffer;
    size_t indexCount;
};

typedef std::shared_ptr<TrackedDeviceModel> TrackedDeviceModelPtr;
struct TrackedHandController
{
    ModelState modelState;
    TrackedDeviceModelPtr deviceModel;

    agpu_buffer_ref modelStateBuffer;
    agpu_texture_ref modelTexture;
    agpu_shader_resource_binding_ref modelStateBinding;

    uint64_t touchedButtons;
    uint64_t pressedButtons;
    Vector2 joysticAxisState;
    Vector2 trackpadAxisState;
    Vector2 triggerAxisState;
    void convertState(const agpu_vr_controller_state &sourceState);
    void convertAxis(int index, const agpu_vr_controller_axis_state &sourceState);

};

class Molevis
{
public:
    Molevis() = default;
    ~Molevis() = default;

    int mainStart(int argc, const char *argv[]);
    void createIntermediateTexturesAndFramebuffer();
    void createPointerModel();

    Random randColor;
    std::unordered_map<std::string, Vector4> atomTypeColorMap;
    PeriodicTable periodicTable;

    void initializeAtomColorConventions();
    void loadPeriodicTable();

    Vector4 getOrCreateColorForAtomType(const std::string &type);

    void convertChemfileFrame(chemfiles::Frame &frame);
    void generateTestDataset();
    void generateRandomDataset(size_t atomsToGenerate, size_t bondsToGenerate);

    DAABox atomsBoundingBox;

    void computeAtomsBoundingBox();

    std::string readWholeFile(const std::string &fileName);
    agpu_shader_ref compileShaderWithSourceFile(const std::string &sourceFileName, agpu_shader_type type);
    agpu_shader_ref compileShaderWithCommonSourceFile(const std::string &commonSourceFile, const std::string &sourceFileName, agpu_shader_type type);

    agpu_shader_ref compileShaderWithSource(const std::string &name, const std::string &source, agpu_shader_type type);

    agpu_pipeline_state_ref finishBuildingPipeline(const agpu_pipeline_builder_ref &builder);

    agpu_pipeline_state_ref compileAndBuildComputeShaderPipelineWithSourceFile(const std::string &commonSourceFilename, const std::string &filename);

    agpu_pipeline_state_ref finishBuildingComputePipeline(const agpu_compute_pipeline_builder_ref &builder);

    void processEvents();
    void processEvent(const SDL_Event &event);

    void recreateSwapChain();

    void onKeyDown(const SDL_KeyboardEvent &event);
    void onMouseMotion(const SDL_MouseMotionEvent &event);

    void onMouseWheel(const SDL_MouseWheelEvent &event);

    void drawRectangle(const Vector2 &position, const Vector2 &size, const Vector4 &color);

    Vector2 drawGlyph(char c, const Vector2 &position, const Vector4 &color);
    Vector2 drawString(const std::string &string, const Vector2 &position, const Vector4 &color);

    float currentLayoutRowX = 0;
    float currentLayoutRowY = 0;
    float currentLayoutX = 0;
    float currentLayoutY = 0;
    
    void beginLayout(float x = 5, float y = 5);
    void advanceLayoutRow();

    void simulateIterationInCPU(double timestep);
    void simulationThreadEntry();
    void startSimulationThread();

    TrackedDeviceModelPtr loadDeviceModel(agpu_vr_render_model *agpuModel);
    agpu_texture_ref loadVRModelTexture(agpu_vr_render_model_texture *vrTexture);

    void updateAndRender(float delta);
    void findHighlightedAtom(const Ray &ray);
    void updateVRState();
    void emitRenderCommands();
    void emitCommandsForEyeRendering(bool isRightEye);

    void swapBuffers();

    agpu_texture_ref loadTexture(const char *fileName, bool nonColorData);
    
    SDL_Window *window = nullptr;
    std::atomic_bool isQuitting = false;

    agpu_texture_format colorBufferFormat = AGPU_TEXTURE_FORMAT_R16G16B16A16_FLOAT;
    agpu_texture_format swapChainColorBufferFormat = AGPU_TEXTURE_FORMAT_B8G8R8A8_UNORM_SRGB;
    agpu_texture_format depthBufferFormat = AGPU_TEXTURE_FORMAT_D32_FLOAT;

    agpu_device_ref device;
    agpu_vr_system_ref vrSystem;
    agpu_command_queue_ref commandQueue;
    agpu_renderpass_ref mainRenderPass;
    agpu_renderpass_ref outputRenderPass;
    agpu_shader_signature_ref shaderSignature;
    agpu_command_allocator_ref commandAllocator;
    agpu_command_list_ref commandList;
    agpu_swap_chain_create_info currentSwapChainCreateInfo;
    agpu_swap_chain_ref swapChain;

    int framebufferDisplayWidth = -1;
    int framebufferDisplayHeight = -1;
    int vrDisplayWidth = -1;
    int vrDisplayHeight = -1;
    agpu_texture_ref depthStencilTexture;
    agpu_texture_ref hdrTargetTexture;
    agpu_framebuffer_ref hdrTargetFramebuffer;

    agpu_texture_ref leftEyeTexture;
    agpu_framebuffer_ref leftEyeFramebuffer;

    agpu_texture_ref rightEyeTexture;
    agpu_framebuffer_ref rightEyeFramebuffer;

    agpu_shader_ref uiElementVertex;
    agpu_shader_ref uiElementFragment;
    agpu_pipeline_state_ref uiPipeline;

    agpu_shader_ref screenQuadVertex;
    agpu_shader_ref filmicTonemappingFragment;
    agpu_pipeline_state_ref filmicTonemappingPipeline;

    agpu_shader_ref passthroughFragment;
    agpu_pipeline_state_ref passthroughPipeline;

    agpu_shader_ref sideBySideFragment;
    agpu_pipeline_state_ref sideBySidePipeline;

    TrackedHandController handControllers[2];
    agpu_vertex_layout_ref modelVertexLayout;
    agpu_pipeline_state_ref modelPipelineState;

    agpu_vertex_layout_ref pointerModelVertexLayout;
    agpu_pipeline_state_ref pointerModelPipelineState;
    TrackedDeviceModelPtr pointerModel;

    agpu_sampler_ref sampler;
    agpu_shader_resource_binding_ref samplersBinding;

    agpu_buffer_ref leftEyeCameraStateUniformBuffer;
    agpu_buffer_ref rightEyeCameraStateUniformBuffer;
    agpu_buffer_ref uiDataBuffer;
    agpu_shader_resource_binding_ref leftEyeScreenStateBinding;
    agpu_shader_resource_binding_ref rightEyeScreenStateBinding;

    agpu_shader_ref screenBoundQuadVertex;
    agpu_shader_ref atomDrawFragment;
    agpu_pipeline_state_ref atomDrawPipeline;

    agpu_shader_ref bondDrawVertex;
    agpu_shader_ref bondDrawFragment;
    agpu_pipeline_state_ref bondDrawPipeline;
    agpu_pipeline_state_ref bondXRayDrawPipeline;
    bool bondXRay = false;

    agpu_pipeline_state_ref floorGridDrawPipeline;

    agpu_buffer_ref atomBoundQuadBuffer;
    agpu_pipeline_state_ref atomScreenQuadBufferComputationPipeline;
    agpu_shader_resource_binding_ref atomBoundQuadBufferBinding;

    std::vector<AtomDescription> atomDescriptions; 
    std::vector<AtomBondDescription> atomBondDescriptions; 
    std::vector<AtomSimulationState> simulationAtomState;

    std::mutex renderingAtomStateMutex;
    std::vector<AtomState> renderingAtomState;
    bool renderingAtomStateDirty = true;

    std::mutex simulationThreadStateMutex;
    std::condition_variable simulationThreadStateConditionChanged;
    std::thread simulationThread;

    agpu_buffer_ref atomDescriptionBuffer;
    agpu_buffer_ref atomBondDescriptionBuffer;
    agpu_buffer_ref atomStateFrontBuffer;
    agpu_shader_resource_binding_ref atomFrontBufferBinding;

    agpu_texture_ref bitmapFont;
    float bitmapFontScale = 1.5;
    float bitmapFontInverseWidth = 0;
    float bitmapFontInverseHeight = 0;
    int bitmapFontGlyphWidth = 7;
    int bitmapFontGlyphHeight = 9;
    int bitmapFontColumns = 16;

    CameraState cameraState;
    CameraState hmdCameraState = cameraState;
    CameraState leftEyeCameraState = hmdCameraState;
    CameraState rightEyeCameraState = hmdCameraState;

    Matrix3x3 cameraMatrix = Matrix3x3::identity();
    Vector3 cameraAngle = Vector3{0, 0, 0};
    Vector3 cameraTranslation = Vector3{0, 0.5, 2};
    Frustum cameraViewFrustum;
    Frustum cameraWorldFrustum;
    Frustum cameraAtomFrustum;

    int32_t currentHighlightedAtom = -1;

    size_t UIElementQuadBufferMaxCapacity = 4192;
    std::vector<UIElementQuad> uiElementQuadBuffer;

    bool isStereo = false;
    bool isVirtualReality = false;

    Vector3 modelPosition = Vector3(0, 0, 0);
    float modelScaleFactor = 0.1f;

    std::atomic_bool isSimulating = true;
    std::atomic_int simulationIteration = 0;

    bool hasWheelEvent = false;
    bool hasHandledWheelEvent = false;
    int wheelDelta = 0;

    bool hasLeftDragEvent = false;
    bool hasHandledLeftDragEvent = false;
    int leftDragStartX = 0;
    int leftDragStartY = 0;
    int leftDragDeltaX = 0;
    int leftDragDeltaY = 0;

    bool hasRightDragEvent = false;
    bool hasHandledRightDragEvent = false;
    int rightDragStartX = 0;
    int rightDragStartY = 0;
    int rightDragDeltaX = 0;
    int rightDragDeltaY = 0;

    int mousePositionX = 0;
    int mousePositionY = 0;

    int displayWidth = 640;
    int displayHeight = 480;
};

#endif //MOLEVIS_HPP