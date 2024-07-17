#include "SDL.h"
#include "SDL_syswm.h"
#include "AGPU/agpu.hpp"
#include "AtomBondDescription.hpp"
#include "AtomDescription.hpp"
#include "AtomState.hpp"
#include "CameraState.hpp"
#include "Vector2.hpp"
#include "Vector3.hpp"
#include "Vector4.hpp"
#include "Matrix4x4.hpp"
#include "PushConstants.hpp"
#include <stdint.h>
#include <stdio.h>
#include <memory>
#include <vector>
#include <string>
#include <random>
#include <time.h>

int64_t getMicroseconds()
{
    timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return int64_t(ts.tv_sec*1000000) + int64_t(ts.tv_nsec/1000);
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
        return std::uniform_real_distribution<> (min, max)(rand);
    }

    Vector3 randVector3(const Vector3 &min, const Vector3 &max)
    {
        return Vector3{randFloat(min.x, max.x), randFloat(min.y, max.y), randFloat(min.z, max.z)};
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

class Mollevis
{
public:
    Mollevis() = default;
    ~Mollevis() = default;

    int main(int argc, const char *argv[])
    {
        bool vsyncDisabled = false;
        bool debugLayerEnabled = false;
    #ifdef _DEBUG
        debugLayerEnabled= true;
    #endif
        agpu_uint platformIndex = 0;
        agpu_uint gpuIndex = 0;
        for (int i = 1; i < argc; ++i)
        {
            std::string arg = argv[i];
            if (arg == "-no-vsync")
            {
                vsyncDisabled = true;
            }
            else if (arg == "-platform")
            {
                platformIndex = agpu_uint(atoi(argv[++i]));
            }
            else if (arg == "-gpu")
            {
                gpuIndex = agpu_uint(atoi(argv[++i]));
            }
            else if (arg == "-debug")
            {
                debugLayerEnabled = true;
            }
        }

        generateRandomDataset(1000, 200);

        // Get the platform.
        agpu_uint numPlatforms;
        agpuGetPlatforms(0, nullptr, &numPlatforms);
        if (numPlatforms == 0)
        {
            fprintf(stderr, "No agpu platforms are available.\n");
            return 1;
        }
        else if (platformIndex >= numPlatforms)
        {
            fprintf(stderr, "Selected platform index is not available.\n");
            return 1;
        }

        std::vector<agpu_platform*> platforms;
        platforms.resize(numPlatforms);
        agpuGetPlatforms(numPlatforms, &platforms[0], nullptr);
        auto platform = platforms[platformIndex];

        printf("Choosen platform: %s\n", agpuGetPlatformName(platform));

        SDL_SetHint(SDL_HINT_NO_SIGNAL_HANDLERS, "1");
        SDL_Init(SDL_INIT_VIDEO);

        window = SDL_CreateWindow("Mollevis", SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED, cameraState.screenWidth, cameraState.screenHeight, SDL_WINDOW_SHOWN | SDL_WINDOW_RESIZABLE);
        if(!window)
        {
            fprintf(stderr, "Failed to create window.\n");
            return 1;
        }

        // Get the window info.
        SDL_SysWMinfo windowInfo;
        SDL_VERSION(&windowInfo.version);
        SDL_GetWindowWMInfo(window, &windowInfo);

        // Open the device
        agpu_device_open_info openInfo = {};
        openInfo.gpu_index = gpuIndex;
        openInfo.debug_layer = debugLayerEnabled;
        memset(&currentSwapChainCreateInfo, 0, sizeof(currentSwapChainCreateInfo));
        switch(windowInfo.subsystem)
        {
    #if defined(SDL_VIDEO_DRIVER_WINDOWS)
        case SDL_SYSWM_WINDOWS:
            currentSwapChainCreateInfo.window = (agpu_pointer)windowInfo.info.win.window;
            break;
    #endif
    #if defined(SDL_VIDEO_DRIVER_X11)
        case SDL_SYSWM_X11:
            openInfo.display = (agpu_pointer)windowInfo.info.x11.display;
            currentSwapChainCreateInfo.window = (agpu_pointer)(uintptr_t)windowInfo.info.x11.window;
            break;
    #endif
    #if defined(SDL_VIDEO_DRIVER_COCOA)
        case SDL_SYSWM_COCOA:
            currentSwapChainCreateInfo.window = (agpu_pointer)windowInfo.info.cocoa.window;
            break;
    #endif
        default:
            fprintf(stderr, "Unsupported window system\n");
            return -1;
        }

        currentSwapChainCreateInfo.colorbuffer_format = colorBufferFormat;
        currentSwapChainCreateInfo.depth_stencil_format = depthBufferFormat;
        currentSwapChainCreateInfo.width = cameraState.screenWidth;
        currentSwapChainCreateInfo.height = cameraState.screenHeight;
        currentSwapChainCreateInfo.buffer_count = 3;
        currentSwapChainCreateInfo.flags = AGPU_SWAP_CHAIN_FLAG_APPLY_SCALE_FACTOR_FOR_HI_DPI;
        if (vsyncDisabled)
        {
            currentSwapChainCreateInfo.presentation_mode = AGPU_SWAP_CHAIN_PRESENTATION_MODE_MAILBOX;
            currentSwapChainCreateInfo.fallback_presentation_mode = AGPU_SWAP_CHAIN_PRESENTATION_MODE_IMMEDIATE;
        }

        device = platform->openDevice(&openInfo);
        if(!device)
        {
            fprintf(stderr, "Failed to open the device\n");
            return false;
        }

        // Get the default command queue
        commandQueue = device->getDefaultCommandQueue();

        // Create the swap chain.
        swapChain = device->createSwapChain(commandQueue, &currentSwapChainCreateInfo);
        if(!swapChain)
        {
            fprintf(stderr, "Failed to create the swap chain\n");
            return false;
        }

        displayWidth = swapChain->getWidth();
        displayHeight = swapChain->getHeight();

        // Create the render pass
        {
            agpu_renderpass_color_attachment_description colorAttachment = {};
            colorAttachment.format = colorBufferFormat;
            colorAttachment.begin_action = AGPU_ATTACHMENT_CLEAR;
            colorAttachment.end_action = AGPU_ATTACHMENT_KEEP;
            colorAttachment.clear_value.r = 0.5;
            colorAttachment.clear_value.g = 0.5;
            colorAttachment.clear_value.b = 0.5;
            colorAttachment.clear_value.a = 0;
            colorAttachment.sample_count = 1;

            agpu_renderpass_depth_stencil_description depthAttachment = {};
            depthAttachment.format = depthBufferFormat;
            depthAttachment.begin_action = AGPU_ATTACHMENT_CLEAR;
            depthAttachment.end_action = AGPU_ATTACHMENT_KEEP;
            depthAttachment.clear_value.depth = 0.0;
            depthAttachment.sample_count = 1;

            agpu_renderpass_description description = {};
            description.color_attachment_count = 1;
            description.color_attachments = &colorAttachment;
            description.depth_stencil_attachment = &depthAttachment;

            mainRenderPass = device->createRenderPass(&description);
        }

        // Create the shader signature
        {

            auto builder = device->createShaderSignatureBuilder();
            // Sampler
            builder->beginBindingBank(1);
            builder->addBindingBankElement(AGPU_SHADER_BINDING_TYPE_SAMPLER, 1);

            builder->beginBindingBank(1);
            builder->addBindingBankElement(AGPU_SHADER_BINDING_TYPE_UNIFORM_BUFFER, 1); // Screen and UI state
            builder->addBindingBankElement(AGPU_SHADER_BINDING_TYPE_STORAGE_BUFFER, 1); // UI Data
            builder->addBindingBankElement(AGPU_SHADER_BINDING_TYPE_SAMPLED_IMAGE, 1); // Bitmap font

            builder->beginBindingBank(2);
            builder->addBindingBankElement(AGPU_SHADER_BINDING_TYPE_STORAGE_BUFFER, 1); // Atom Description Buffer
            builder->addBindingBankElement(AGPU_SHADER_BINDING_TYPE_STORAGE_BUFFER, 1); // Atom Bond Buffer
            builder->addBindingBankElement(AGPU_SHADER_BINDING_TYPE_STORAGE_BUFFER, 1); // Atom Front State Buffer
            builder->addBindingBankElement(AGPU_SHADER_BINDING_TYPE_STORAGE_BUFFER, 1); // Atom Back State Buffer

            builder->beginBindingBank(2);
            builder->addBindingBankElement(AGPU_SHADER_BINDING_TYPE_STORAGE_BUFFER, 1); // Bounding Quad Buffer

            builder->addBindingConstant(); // Timestep
            builder->addBindingConstant(); // Atom count
            builder->addBindingConstant(); // Bond count

            shaderSignature = builder->build();
            if(!shaderSignature)
                return 1;
        }

        // Samplers binding
        {
            agpu_sampler_description samplerDesc = {};
            samplerDesc.address_u = AGPU_TEXTURE_ADDRESS_MODE_CLAMP;
            samplerDesc.address_v = AGPU_TEXTURE_ADDRESS_MODE_CLAMP;
            samplerDesc.address_w = AGPU_TEXTURE_ADDRESS_MODE_CLAMP;
            samplerDesc.filter = AGPU_FILTER_MIN_LINEAR_MAG_LINEAR_MIPMAP_NEAREST;
            sampler = device->createSampler(&samplerDesc);
            if(!sampler)
            {
                fprintf(stderr, "Failed to create the sampler.\n");
                return 1;
            }

            samplersBinding = shaderSignature->createShaderResourceBinding(0);
            samplersBinding->bindSampler(0, sampler);
        }

        // Screen and UI State buffer
        {
            agpu_buffer_description desc = {};
            desc.size = (sizeof(CameraState) + 255) & (-256);
            desc.heap_type = AGPU_MEMORY_HEAP_TYPE_HOST_TO_DEVICE;
            desc.usage_modes = agpu_buffer_usage_mask(AGPU_COPY_DESTINATION_BUFFER | AGPU_UNIFORM_BUFFER);
            desc.main_usage_mode = AGPU_UNIFORM_BUFFER;
	        desc.mapping_flags = AGPU_MAP_DYNAMIC_STORAGE_BIT;
            cameraStateUniformBuffer = device->createBuffer(&desc, nullptr);
        }

        {
            uiElementQuadBuffer.reserve(UIElementQuadBufferMaxCapacity);
            agpu_buffer_description desc = {};
            desc.size = (sizeof(UIElementQuad)*UIElementQuadBufferMaxCapacity + 255) & (-256);
            desc.heap_type = AGPU_MEMORY_HEAP_TYPE_HOST_TO_DEVICE;
            desc.usage_modes = agpu_buffer_usage_mask(AGPU_COPY_DESTINATION_BUFFER | AGPU_STORAGE_BUFFER);
            desc.main_usage_mode = AGPU_STORAGE_BUFFER;
	        desc.mapping_flags = AGPU_MAP_DYNAMIC_STORAGE_BIT;
            uiDataBuffer = device->createBuffer(&desc, nullptr);
        }

        bitmapFont = loadTexture("assets/textures/pixel_font_basic_latin_ascii.bmp", false);
        if(!bitmapFont)
        {
            fprintf(stderr, "Failed to load the bitmap font.");
            return 1;
        }
        {
            agpu_texture_description desc;
            bitmapFont->getDescription(&desc);
            bitmapFontInverseWidth = 1.0 / desc.width;
            bitmapFontInverseHeight = 1.0 / desc.height;
        }

        // Atom description buffer
        {
            agpu_buffer_description desc = {};
            desc.size = (sizeof(AtomDescription)*std::max(atomDescriptions.size(), size_t(1024)) + 255) & (-256);
            desc.heap_type = AGPU_MEMORY_HEAP_TYPE_DEVICE_LOCAL;
            desc.usage_modes = agpu_buffer_usage_mask(AGPU_COPY_DESTINATION_BUFFER | AGPU_STORAGE_BUFFER);
            desc.main_usage_mode = AGPU_STORAGE_BUFFER;
	        desc.mapping_flags = AGPU_MAP_DYNAMIC_STORAGE_BIT;
            atomDescriptionBuffer = device->createBuffer(&desc, nullptr);
            atomDescriptionBuffer->uploadBufferData(0, sizeof(AtomDescription)*atomDescriptions.size(), atomDescriptions.data());
        }

        // Atom bond description buffer
        {
            agpu_buffer_description desc = {};
            desc.size = (sizeof(AtomBondDescription)*std::max(atomBondDescriptions.size(), size_t(1024)) + 255) & (-256);
            desc.heap_type = AGPU_MEMORY_HEAP_TYPE_DEVICE_LOCAL;
            desc.usage_modes = agpu_buffer_usage_mask(AGPU_COPY_DESTINATION_BUFFER | AGPU_STORAGE_BUFFER);
            desc.main_usage_mode = AGPU_STORAGE_BUFFER;
	        desc.mapping_flags = AGPU_MAP_DYNAMIC_STORAGE_BIT;
            atomBondDescriptionBuffer = device->createBuffer(&desc, nullptr);
            atomBondDescriptionBuffer->uploadBufferData(0, sizeof(AtomBondDescription)*atomBondDescriptions.size(), atomBondDescriptions.data());
        }

        // Atom state buffers
        {
            agpu_buffer_description desc = {};
            desc.size = (sizeof(AtomState)*std::max(initialAtomStates.size(), size_t(1024)) + 255) & (-256);
            desc.heap_type = AGPU_MEMORY_HEAP_TYPE_DEVICE_LOCAL;
            desc.usage_modes = agpu_buffer_usage_mask(AGPU_COPY_DESTINATION_BUFFER | AGPU_STORAGE_BUFFER);
            desc.main_usage_mode = AGPU_STORAGE_BUFFER;
	        desc.mapping_flags = AGPU_MAP_DYNAMIC_STORAGE_BIT;
            
            size_t uploadSize = sizeof(AtomState)*initialAtomStates.size();
            atomStateFrontBuffer = device->createBuffer(&desc, nullptr);
            atomStateFrontBuffer->uploadBufferData(0, uploadSize, initialAtomStates.data());

            atomStateBackBuffer = device->createBuffer(&desc, nullptr);
            atomStateBackBuffer->uploadBufferData(0, uploadSize, initialAtomStates.data());
        }

        atomFrontBufferBinding = shaderSignature->createShaderResourceBinding(2);
        atomFrontBufferBinding->bindStorageBuffer(0, atomDescriptionBuffer);
        atomFrontBufferBinding->bindStorageBuffer(1, atomBondDescriptionBuffer);
        atomFrontBufferBinding->bindStorageBuffer(2, atomStateFrontBuffer);
        atomFrontBufferBinding->bindStorageBuffer(3, atomStateBackBuffer);

        atomBackBufferBinding = shaderSignature->createShaderResourceBinding(2);
        atomBackBufferBinding->bindStorageBuffer(0, atomDescriptionBuffer);
        atomBackBufferBinding->bindStorageBuffer(1, atomBondDescriptionBuffer);
        atomBackBufferBinding->bindStorageBuffer(2, atomStateBackBuffer);
        atomBackBufferBinding->bindStorageBuffer(3, atomStateFrontBuffer);

        // Atom bounding quad buffer
        {
            agpu_buffer_description desc = {};
            size_t requiredCapacity = std::max(atomDescriptions.size(), size_t(1024));
            requiredCapacity = (requiredCapacity + 31) / 32 * 32;

            desc.size = (32*requiredCapacity + 255) & (-256);
            desc.heap_type = AGPU_MEMORY_HEAP_TYPE_DEVICE_LOCAL;
            desc.usage_modes = agpu_buffer_usage_mask(AGPU_STORAGE_BUFFER);
            desc.main_usage_mode = AGPU_STORAGE_BUFFER;

            atomBoundQuadBuffer = device->createBuffer(&desc, nullptr);

            atomBoundQuadBufferBinding = shaderSignature->createShaderResourceBinding(3);
            atomBoundQuadBufferBinding->bindStorageBuffer(0, atomBoundQuadBuffer);
        }

        // Simulation pipelines
        simulationResetTimeStepPipeline = compileAndBuildComputeShaderPipelineWithSourceFile("assets/shaders/simulationResetTimeStep.glsl");
        simulationLennardJonesPipeline = compileAndBuildComputeShaderPipelineWithSourceFile("assets/shaders/simulationLennardJones.glsl");
        simulationIntegratePipeline = compileAndBuildComputeShaderPipelineWithSourceFile("assets/shaders/simulationIntegrate.glsl");

        // Atom screen quad computation shader
        atomScreenQuadBufferComputationPipeline = compileAndBuildComputeShaderPipelineWithSourceFile("assets/shaders/atomScreenQuadComputation.glsl");

        // Atom and bond draw pipeline state.
        screenBoundQuadVertex = compileShaderWithSourceFile("assets/shaders/screenBoundQuadVertex.glsl", AGPU_VERTEX_SHADER);
        atomDrawFragment = compileShaderWithSourceFile("assets/shaders/atomFragment.glsl", AGPU_FRAGMENT_SHADER);
        bondDrawVertex = compileShaderWithSourceFile("assets/shaders/bondVertex.glsl", AGPU_VERTEX_SHADER);
        bondDrawFragment = compileShaderWithSourceFile("assets/shaders/bondFragment.glsl", AGPU_FRAGMENT_SHADER);
        if(!screenBoundQuadVertex || !atomDrawFragment || !bondDrawVertex || !bondDrawFragment)
            return 1;

        {
            auto builder = device->createPipelineBuilder();
            builder->setRenderTargetFormat(0, colorBufferFormat);
            builder->setDepthStencilFormat(depthBufferFormat);
            builder->setShaderSignature(shaderSignature);
            builder->attachShader(screenBoundQuadVertex);
            builder->attachShader(atomDrawFragment);
            builder->setPrimitiveType(AGPU_TRIANGLE_STRIP);
            builder->setDepthState(true, true, AGPU_GREATER_EQUAL);
            atomDrawPipeline = finishBuildingPipeline(builder);
        }

        {
            auto builder = device->createPipelineBuilder();
            builder->setRenderTargetFormat(0, colorBufferFormat);
            builder->setDepthStencilFormat(depthBufferFormat);
            builder->setShaderSignature(shaderSignature);
            builder->attachShader(bondDrawVertex);
            builder->attachShader(bondDrawFragment);
            builder->setPrimitiveType(AGPU_TRIANGLE_STRIP);
            builder->setDepthState(true, true, AGPU_GREATER_EQUAL);
            bondDrawPipeline = finishBuildingPipeline(builder);
        }
        // Data binding
        screenStateBinding = shaderSignature->createShaderResourceBinding(1);
        screenStateBinding->bindUniformBuffer(0, cameraStateUniformBuffer);
        screenStateBinding->bindStorageBuffer(1, uiDataBuffer);
        screenStateBinding->bindSampledTextureView(2, bitmapFont->getOrCreateFullView());

        // UI pipeline state.
        uiElementVertex = compileShaderWithSourceFile("assets/shaders/uiElementVertex.glsl", AGPU_VERTEX_SHADER);
        uiElementFragment = compileShaderWithSourceFile("assets/shaders/uiElementFragment.glsl", AGPU_FRAGMENT_SHADER);

        if(!uiElementVertex || !uiElementFragment)
            return 1;

        {
            auto builder = device->createPipelineBuilder();
            builder->setRenderTargetFormat(0, colorBufferFormat);
            builder->setShaderSignature(shaderSignature);
            builder->attachShader(uiElementVertex);
            builder->attachShader(uiElementFragment);
            builder->setBlendFunction(-1,
                AGPU_BLENDING_ONE, AGPU_BLENDING_INVERTED_SRC_ALPHA, AGPU_BLENDING_OPERATION_ADD,
                AGPU_BLENDING_ONE, AGPU_BLENDING_INVERTED_SRC_ALPHA, AGPU_BLENDING_OPERATION_ADD
            );
            builder->setBlendState(-1, true);
            builder->setPrimitiveType(AGPU_TRIANGLE_STRIP);
            uiPipeline = finishBuildingPipeline(builder);
        }

        // Create the command allocator and command list
        commandAllocator = device->createCommandAllocator(AGPU_COMMAND_LIST_TYPE_DIRECT, commandQueue);
        commandList = device->createCommandList(AGPU_COMMAND_LIST_TYPE_DIRECT, commandAllocator, nullptr);
        commandList->close();

        // Main loop
        auto oldTime = getMicroseconds();
        while(!isQuitting)
        {
            auto newTime = getMicroseconds();
            auto deltaTime = newTime - oldTime;
            oldTime = newTime;

            processEvents();
            updateAndRender(deltaTime * 1.0e-6f);
        }

        commandQueue->finishExecution();
        swapChain.reset();
        commandQueue.reset();

        SDL_DestroyWindow(window);
        SDL_Quit();
        return 0;
    }

    void generateRandomDataset(size_t atomsToGenerate, size_t bondsToGenerate)
    {
        Random rand;
        atomDescriptions.reserve(atomsToGenerate);
        initialAtomStates.reserve(atomsToGenerate);

        for(size_t i = 0; i < atomsToGenerate; ++i)
        {
            auto description = AtomDescription{};
            auto state = AtomState{};

            description.lennardJonesEpsilon = rand.randFloat(1, 5);
            description.lennardJonesSigma = rand.randFloat(1, 5);
            description.radius = rand.randFloat(0.5, 2);
            description.color = rand.randVector4(Vector4{0.1, 0.1, 0.1, 1.0}, Vector4{0.8, 0.8, 0.8, 1.0});
            state.position = rand.randVector3(-100, 100);

            atomDescriptions.push_back(description);
            initialAtomStates.push_back(state);
        }

        for(size_t i = 0; i < bondsToGenerate; ++i)
        {
            auto firstAtomIndex = rand.randUInt(atomDescriptions.size());
            auto secondAtomIndex = firstAtomIndex;
            while(firstAtomIndex == secondAtomIndex)
                secondAtomIndex = rand.randUInt(atomDescriptions.size());

            auto description = AtomBondDescription{};
            description.firstAtomIndex = firstAtomIndex;
            description.secondAtomIndex = secondAtomIndex;
            description.morseEquilibriumDistance = rand.randFloat(5, 20);
            description.morseWellDepth = 1;
            description.thickness = rand.randFloat(0.1, 0.4);
            description.color = rand.randVector4(Vector4{0.1, 0.1, 0.1, 1.0}, Vector4{0.8, 0.8, 0.8, 1.0});
            atomBondDescriptions.push_back(description);
        }
    }

    std::string readWholeFile(const std::string &fileName)
    {
        FILE *file = fopen(fileName.c_str(), "rb");
        if(!file)
        {
            fprintf(stderr, "Failed to open file %s\n", fileName.c_str());
            return std::string();
        }

        // Allocate the data.
        std::vector<char> data;
        fseek(file, 0, SEEK_END);
        data.resize(ftell(file));
        fseek(file, 0, SEEK_SET);

        // Read the file
        if(fread(&data[0], data.size(), 1, file) != 1)
        {
            fprintf(stderr, "Failed to read file %s\n", fileName.c_str());
            fclose(file);
            return std::string();
        }

        fclose(file);
        return std::string(data.begin(), data.end());
    }

    agpu_shader_ref compileShaderWithSourceFile(const std::string &sourceFileName, agpu_shader_type type)
    {
        return compileShaderWithSource(sourceFileName, readWholeFile(sourceFileName), type);
    }

    agpu_shader_ref compileShaderWithSource(const std::string &name, const std::string &source, agpu_shader_type type)
    {
        if(source.empty())
            return nullptr;

        // Create the shader compiler.
        agpu_offline_shader_compiler_ref shaderCompiler = device->createOfflineShaderCompiler();
        shaderCompiler->setShaderSource(AGPU_SHADER_LANGUAGE_VGLSL, type, source.c_str(), (agpu_string_length)source.size());
        auto errorCode = agpuCompileOfflineShader(shaderCompiler.get(), AGPU_SHADER_LANGUAGE_DEVICE_SHADER, nullptr);
        if(errorCode)
        {
            auto logLength = shaderCompiler->getCompilationLogLength();
            std::unique_ptr<char[]> logBuffer(new char[logLength+1]);
            shaderCompiler->getCompilationLog(logLength+1, logBuffer.get());
            fprintf(stderr, "Compilation error of '%s':%s\n", name.c_str(), logBuffer.get());
            return nullptr;
        }

        // Create the shader and compile it.
        return shaderCompiler->getResultAsShader();
    }

    agpu_pipeline_state_ref finishBuildingPipeline(const agpu_pipeline_builder_ref &builder)
    {
        auto pipeline = builder->build();
        if(!pipeline)
        {
            fprintf(stderr, "Failed to build pipeline.\n");
        }
        return pipeline;
    }

    agpu_pipeline_state_ref compileAndBuildComputeShaderPipelineWithSourceFile(const std::string &filename)
    {
        auto shader = compileShaderWithSourceFile(filename, AGPU_COMPUTE_SHADER);
        auto builder = device->createComputePipelineBuilder();
        builder->setShaderSignature(shaderSignature);
        builder->attachShader(shader);
        return finishBuildingComputePipeline(builder);
    }

    agpu_pipeline_state_ref finishBuildingComputePipeline(const agpu_compute_pipeline_builder_ref &builder)
    {
        auto pipeline = builder->build();
        if(!pipeline)
        {
            fprintf(stderr, "Failed to build pipeline.\n");
        }
        return pipeline;
    }

    void processEvents()
    {
        // Reset the event data.
        hasWheelEvent = false;
        hasHandledWheelEvent = false;
        wheelDelta = 0;

        hasLeftDragEvent = false;
        hasHandledLeftDragEvent = false;
        leftDragStartX = 0;
        leftDragStartY = 0;
        leftDragDeltaX = 0;
        leftDragDeltaY = 0;

        hasRightDragEvent = false;
        hasHandledRightDragEvent = false;
        rightDragStartX = 0;
        rightDragStartY = 0;
        rightDragDeltaX = 0;
        rightDragDeltaY = 0;

        // Poll and process the SDL events.
        SDL_Event event;
        while(SDL_PollEvent(&event))
            processEvent(event);
    }

    void processEvent(const SDL_Event &event)
    {
        switch(event.type)
        {
        case SDL_QUIT:
            isQuitting = true;
            break;
        case SDL_KEYDOWN:
            onKeyDown(event.key);
            break;
        case SDL_MOUSEMOTION:
            onMouseMotion(event.motion);
            break;
        case SDL_MOUSEWHEEL:
            onMouseWheel(event.wheel);
            break;
        case SDL_WINDOWEVENT:
            {
                switch(event.window.event)
                {
                case SDL_WINDOWEVENT_RESIZED:
                case SDL_WINDOWEVENT_SIZE_CHANGED:
                    recreateSwapChain();
                    break;
                default:
                    break;
                }
            }
            break;
        default:
            break;
        }
    }

    void recreateSwapChain()
    {
        int w, h;
        SDL_GetWindowSize(window, &w, &h);
        cameraState.screenWidth = w;
        cameraState.screenHeight = h;

        device->finishExecution();
        auto newSwapChainCreateInfo = currentSwapChainCreateInfo;
        newSwapChainCreateInfo.width = w;
        newSwapChainCreateInfo.height = h;
        newSwapChainCreateInfo.old_swap_chain = swapChain.get();
        swapChain = device->createSwapChain(commandQueue, &newSwapChainCreateInfo);

        displayWidth = swapChain->getWidth();
        displayHeight = swapChain->getHeight();
        if(swapChain)
            currentSwapChainCreateInfo = newSwapChainCreateInfo;
    }

    void onKeyDown(const SDL_KeyboardEvent &event)
    {
        switch(event.keysym.sym)
        {
        case SDLK_ESCAPE:
            isQuitting = true;
            break;
        default:
            break;
        }
    }

    void onMouseMotion(const SDL_MouseMotionEvent &event)
    {
        if(event.state & SDL_BUTTON_LMASK)
        {
            hasLeftDragEvent = true;
            leftDragStartX = event.x;
            leftDragStartY = event.y;
            leftDragDeltaX = event.xrel;
            leftDragDeltaY = event.yrel;
        }

        if(event.state & SDL_BUTTON_RMASK)
        {
            hasRightDragEvent = true;
            rightDragStartX = event.x;
            rightDragStartY = event.y;
            rightDragDeltaX = event.xrel;
            rightDragDeltaY = event.yrel;
        }
    }

    void onMouseWheel(const SDL_MouseWheelEvent &event)
    {
        hasWheelEvent = true;
        wheelDelta = event.y;
    }

    void drawRectangle(const Vector2 &position, const Vector2 &size, const Vector4 &color)
    {
        UIElementQuad quad = {};
        quad.position = position;
        quad.size = size;
        quad.color = color;

        uiElementQuadBuffer.push_back(quad);
    }

    Vector2 drawGlyph(char c, const Vector2 &position, const Vector4 &color)
    {
        if(c < ' ')
            return Vector2{bitmapFontGlyphWidth*bitmapFontScale, 0.0f};

        UIElementQuad quad = {};
        quad.position = position;
        quad.size = Vector2{bitmapFontGlyphWidth*bitmapFontScale, bitmapFontGlyphHeight*bitmapFontScale};
        quad.color = color;

        if (' ' <= c && c <= 127)
        {
            int index = c - ' ';
            int column = index % bitmapFontColumns;
            int row = index / bitmapFontColumns;
            quad.isGlyph = true;
            quad.fontPosition = Vector2{column * bitmapFontGlyphWidth * bitmapFontInverseWidth, row * bitmapFontGlyphHeight * bitmapFontInverseHeight};
            quad.fontSize = Vector2{bitmapFontGlyphWidth * bitmapFontInverseWidth,  bitmapFontGlyphHeight * bitmapFontInverseHeight};
        }

        uiElementQuadBuffer.push_back(quad);
        return Vector2{bitmapFontGlyphWidth*bitmapFontScale, 0.0f};
    }

    Vector2 drawString(const std::string &string, const Vector2 &position, const Vector4 &color)
    {
        Vector2 totalAdvance = {0, 0};
        for(auto c : string)
            totalAdvance += drawGlyph(c, position + totalAdvance, color);
        return totalAdvance;
    }

    float currentLayoutRowX = 0;
    float currentLayoutRowY = 0;
    float currentLayoutX = 0;
    float currentLayoutY = 0;
    
    void beginLayout(float x = 5, float y = 5)
    {
        currentLayoutX = currentLayoutRowX = x;
        currentLayoutY = currentLayoutRowY = y;
    }

    void advanceLayoutRow()
    {
        currentLayoutRowY += bitmapFontGlyphHeight * bitmapFontScale + 5;
        currentLayoutX = currentLayoutRowX;
        currentLayoutY = currentLayoutRowY;
    }

    void emitSimulationStepCommands()
    {
        // Reset the simulation time step.
        commandList->usePipelineState(simulationResetTimeStepPipeline);
        commandList->dispatchCompute((initialAtomStates.size() + 31)/32, 1, 1);
        commandList->memoryBarrier(AGPU_PIPELINE_STAGE_COMPUTE_SHADER, AGPU_PIPELINE_STAGE_COMPUTE_SHADER, AGPU_ACCESS_SHADER_WRITE, AGPU_ACCESS_SHADER_READ);

        // Accumulate the lennard jones potential energy.
        commandList->usePipelineState(simulationLennardJonesPipeline);
        commandList->dispatchCompute((initialAtomStates.size() + 31)/32, 1, 1);
        commandList->memoryBarrier(AGPU_PIPELINE_STAGE_COMPUTE_SHADER, AGPU_PIPELINE_STAGE_COMPUTE_SHADER, AGPU_ACCESS_SHADER_WRITE, AGPU_ACCESS_SHADER_READ);

        // Integrate simulation time step.
        commandList->usePipelineState(simulationIntegratePipeline);
        commandList->dispatchCompute((initialAtomStates.size() + 31)/32, 1, 1);
        commandList->memoryBarrier(AGPU_PIPELINE_STAGE_COMPUTE_SHADER, AGPU_PIPELINE_STAGE_COMPUTE_SHADER, AGPU_ACCESS_SHADER_WRITE, AGPU_ACCESS_SHADER_READ);

        // Swap the back buffer with the front buffer.
        std::swap(atomBackBufferBinding, atomFrontBufferBinding);
        commandList->useComputeShaderResources(atomFrontBufferBinding);

        ++simulationIteration;
    }

    void updateAndRender(float delta)
    {
        uiElementQuadBuffer.clear();

        // Left drag.
        if(hasLeftDragEvent && !hasHandledLeftDragEvent)
        {
            cameraAngle += Vector3(leftDragDeltaY, leftDragDeltaX, 0) * 0.1/M_PI;
            cameraAngle.x = std::min(std::max(cameraAngle.x, float(-M_PI*0.5)), float(M_PI*0.5));
        }
        cameraMatrix = Matrix3x3::XRotation(cameraAngle.x) * Matrix3x3::YRotation(cameraAngle.y);

        // Right drag.
        if(hasRightDragEvent && !hasHandledRightDragEvent)
        {
            cameraTranslation += cameraMatrix * (Vector3(rightDragDeltaX, -rightDragDeltaY, 0) * 0.1f);
        }

        // Mouse wheel.
        if(hasWheelEvent && !hasHandledWheelEvent)
        {
            cameraTranslation += cameraMatrix * Vector3(0, 0, -wheelDelta);
        }

        char buffer[64];
        snprintf(buffer, sizeof(buffer), "%d Atoms. %d Bonds. Sim iter %05d. Update time %0.3f ms.", int(atomDescriptions.size()), int(atomBondDescriptions.size()), simulationIteration, delta*1000.0);
        drawString(buffer, Vector2{5, 5}, Vector4{0.1, 1.0, 0.1, 1});

        auto cameraInverseMatrix = cameraMatrix.transposed();
        auto cameraInverseTranslation = cameraInverseMatrix * -cameraTranslation;

        cameraState.viewMatrix = Matrix4x4::withMatrix3x3AndTranslation(cameraInverseMatrix, cameraInverseTranslation);
        cameraState.inverseViewMatrix = Matrix4x4::withMatrix3x3AndTranslation(cameraMatrix, cameraTranslation);
        cameraState.projectionMatrix = Matrix4x4::perspective(60.0, float(cameraState.screenWidth)/float(cameraState.screenHeight), cameraState.nearDistance, cameraState.farDistance, device->hasTopLeftNdcOrigin());
        cameraState.inverseProjectionMatrix = cameraState.projectionMatrix.inverse();

        PushConstants pushConstants = {};
        pushConstants.atomCount = atomDescriptions.size();
        pushConstants.bondCount = atomBondDescriptions.size();
        pushConstants.timeStep = simulationTimeStep;

        // Upload the data buffers.
        cameraStateUniformBuffer->uploadBufferData(0, sizeof(cameraState), &cameraState);
        uiDataBuffer->uploadBufferData(0, uiElementQuadBuffer.size() * sizeof(UIElementQuad), uiElementQuadBuffer.data());

        // Build the command list
        commandAllocator->reset();
        commandList->reset(commandAllocator, nullptr);

        auto backBuffer = swapChain->getCurrentBackBuffer();

        commandList->setShaderSignature(shaderSignature);
        commandList->useComputeShaderResources(samplersBinding);
        commandList->useComputeShaderResources(screenStateBinding);
        commandList->useComputeShaderResources(atomFrontBufferBinding);
        commandList->useComputeShaderResources(atomBoundQuadBufferBinding);
        commandList->pushConstants(0, sizeof(pushConstants), &pushConstants);

        // Are we simulating?
        if(isSimulating)
        {
            accumulatedTimeToSimulate += delta;
            if(accumulatedTimeToSimulate >= simulationTimeStep)
            {
                emitSimulationStepCommands();
                accumulatedTimeToSimulate = 0;
            }
        }

        // Screen bounding quad computations.
        commandList->usePipelineState(atomScreenQuadBufferComputationPipeline);
        commandList->dispatchCompute((atomDescriptions.size() + 31)/32, 1, 1);
        commandList->memoryBarrier(AGPU_PIPELINE_STAGE_COMPUTE_SHADER, AGPU_PIPELINE_STAGE_VERTEX_SHADER, AGPU_ACCESS_SHADER_WRITE, AGPU_ACCESS_SHADER_READ);

        commandList->beginRenderPass(mainRenderPass, backBuffer, false);

        commandList->setViewport(0, 0, displayWidth, displayHeight);
        commandList->setScissor(0, 0, displayWidth, displayHeight);

        // Atoms
        commandList->usePipelineState(atomDrawPipeline);
        commandList->useShaderResources(samplersBinding);
        commandList->useShaderResources(screenStateBinding);
        commandList->useShaderResources(atomFrontBufferBinding);
        commandList->useShaderResources(atomBoundQuadBufferBinding);
        commandList->drawArrays(4, atomDescriptions.size(), 0, 0);

        // Bonds
        commandList->usePipelineState(bondDrawPipeline);
        commandList->drawArrays(4, atomBondDescriptions.size(), 0, 0);

        // UI element pipeline
        commandList->usePipelineState(uiPipeline);
        commandList->drawArrays(4, uiElementQuadBuffer.size(), 0, 0);

        // Finish the command list
        commandList->endRenderPass();
        commandList->close();

        // Queue the command list
        commandQueue->addCommandList(commandList);

        swapBuffers();
        commandQueue->finishExecution();
    }

    void swapBuffers()
    {
        auto errorCode = agpuSwapBuffers(swapChain.get());
        if(!errorCode)
            return;

        if(errorCode == AGPU_OUT_OF_DATE)
            recreateSwapChain();
    }

    agpu_texture_ref loadTexture(const char *fileName, bool nonColorData)
    {
        auto surface = SDL_LoadBMP(fileName);
        if (!surface)
            return nullptr;

        auto convertedSurface = SDL_ConvertSurfaceFormat(surface, SDL_PIXELFORMAT_ARGB8888, 0);
        SDL_FreeSurface(surface);
        if (!convertedSurface)
            return nullptr;

        auto format = nonColorData ? AGPU_TEXTURE_FORMAT_B8G8R8A8_UNORM : AGPU_TEXTURE_FORMAT_B8G8R8A8_UNORM_SRGB;
        agpu_texture_description desc = {};
        desc.type = AGPU_TEXTURE_2D;
        desc.format = format;
        desc.width = convertedSurface->w;
        desc.height = convertedSurface->h;
        desc.depth = 1;
        desc.layers = 1;
        desc.miplevels = 1;
        desc.sample_count = 1;
        desc.sample_quality = 0;
        desc.heap_type = AGPU_MEMORY_HEAP_TYPE_DEVICE_LOCAL;
        desc.usage_modes = agpu_texture_usage_mode_mask(AGPU_TEXTURE_USAGE_SAMPLED | AGPU_TEXTURE_USAGE_COPY_DESTINATION);
        desc.main_usage_mode = AGPU_TEXTURE_USAGE_SAMPLED;

        agpu_texture_ref texture = device->createTexture(&desc);
        if (!texture)
        {
            SDL_FreeSurface(convertedSurface);
            return nullptr;
        }

        texture->uploadTextureData(0, 0, convertedSurface->pitch, convertedSurface->pitch*convertedSurface->h, convertedSurface->pixels);
        SDL_FreeSurface(convertedSurface);
        return texture;
    }

    SDL_Window *window = nullptr;
    bool isQuitting = false;

    agpu_texture_format colorBufferFormat = AGPU_TEXTURE_FORMAT_B8G8R8A8_UNORM_SRGB;
    agpu_texture_format depthBufferFormat = AGPU_TEXTURE_FORMAT_D32_FLOAT;

    agpu_device_ref device;
    agpu_command_queue_ref commandQueue;
    agpu_renderpass_ref mainRenderPass;
    agpu_shader_signature_ref shaderSignature;
    agpu_command_allocator_ref commandAllocator;
    agpu_command_list_ref commandList;
    agpu_swap_chain_create_info currentSwapChainCreateInfo;
    agpu_swap_chain_ref swapChain;

    agpu_shader_ref uiElementVertex;
    agpu_shader_ref uiElementFragment;
    agpu_pipeline_state_ref uiPipeline;

    agpu_sampler_ref sampler;
    agpu_shader_resource_binding_ref samplersBinding;

    agpu_buffer_ref cameraStateUniformBuffer;
    agpu_buffer_ref uiDataBuffer;
    agpu_shader_resource_binding_ref screenStateBinding;

    agpu_shader_ref screenBoundQuadVertex;
    agpu_shader_ref atomDrawFragment;
    agpu_pipeline_state_ref atomDrawPipeline;

    agpu_shader_ref bondDrawVertex;
    agpu_shader_ref bondDrawFragment;
    agpu_pipeline_state_ref bondDrawPipeline;

    agpu_buffer_ref atomBoundQuadBuffer;
    agpu_pipeline_state_ref atomScreenQuadBufferComputationPipeline;
    agpu_shader_resource_binding_ref atomBoundQuadBufferBinding;

    std::vector<AtomDescription> atomDescriptions; 
    std::vector<AtomBondDescription> atomBondDescriptions; 
    std::vector<AtomState> initialAtomStates;
    agpu_buffer_ref atomDescriptionBuffer;
    agpu_buffer_ref atomBondDescriptionBuffer;
    agpu_buffer_ref atomStateFrontBuffer;
    agpu_buffer_ref atomStateBackBuffer;
    agpu_shader_resource_binding_ref atomFrontBufferBinding;
    agpu_shader_resource_binding_ref atomBackBufferBinding;

    agpu_pipeline_state_ref simulationResetTimeStepPipeline;
    agpu_pipeline_state_ref simulationLennardJonesPipeline;
    agpu_pipeline_state_ref simulationIntegratePipeline;

    agpu_texture_ref bitmapFont;
    float bitmapFontScale = 1.5;
    float bitmapFontInverseWidth = 0;
    float bitmapFontInverseHeight = 0;
    int bitmapFontGlyphWidth = 7;
    int bitmapFontGlyphHeight = 9;
    int bitmapFontColumns = 16;

    CameraState cameraState;
    Matrix3x3 cameraMatrix = Matrix3x3::identity();
    Vector3 cameraAngle = Vector3{0, 0, 0};
    Vector3 cameraTranslation = Vector3{0, 0, 5};

    size_t UIElementQuadBufferMaxCapacity = 4192;
    std::vector<UIElementQuad> uiElementQuadBuffer;

    bool isSimulating = true;
    float simulationTimeStep = 1.0f/60.0f;
    float accumulatedTimeToSimulate = 0.0f;
    int simulationIteration = 0;

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

    int displayWidth = 640;
    int displayHeight = 480;
};

int main(int argc, const char *argv[])
{
    return Mollevis().main(argc, argv);
}
