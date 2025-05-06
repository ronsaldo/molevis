

#include "Molevis.hpp"
#include "Sphere.hpp"
#include "PushConstants.hpp"

agpu_vertex_attrib_description VRRenderModelVertexDesc[] = {
    {0, 0, AGPU_TEXTURE_FORMAT_R32G32B32_FLOAT, offsetof(agpu_vr_render_model_vertex, position), 0},
    {0, 1, AGPU_TEXTURE_FORMAT_R32G32B32_FLOAT, offsetof(agpu_vr_render_model_vertex, normal), 0},
    {0, 2, AGPU_TEXTURE_FORMAT_R32G32_FLOAT, offsetof(agpu_vr_render_model_vertex, texcoord), 0},
};

const int VRRenderModelVertexDescSize = sizeof(VRRenderModelVertexDesc) / sizeof(VRRenderModelVertexDesc[0]);

agpu_vertex_attrib_description VRPointerModelVertexDesc[] = {
    {0, 0, AGPU_TEXTURE_FORMAT_R32G32B32_FLOAT, 0, 0},
};

const int VRPointerModelVertexDescSize = sizeof(VRPointerModelVertexDesc) / sizeof(VRPointerModelVertexDesc[0]);

void
TrackedHandController::convertState(const agpu_vr_controller_state &sourceState)
{
    touchedButtons = sourceState.buttons_touched;
    pressedButtons = sourceState.buttons_pressed;
    convertAxis(0, sourceState.axis0);
    convertAxis(1, sourceState.axis1);
    convertAxis(2, sourceState.axis2);
    convertAxis(3, sourceState.axis3);
    convertAxis(4, sourceState.axis4);
}

void
TrackedHandController::convertAxis(int index, const agpu_vr_controller_axis_state &sourceState)
{
    switch(sourceState.type)
    {
    case AGPU_VR_CONTROLLER_AXIS_TRACK_PAD:
        trackpadAxisState = Vector2(sourceState.x, sourceState.y);
        //printf("%d: trackpadAxisState %f %f\n", index, trackpadAxisState.x, trackpadAxisState.y);
        break;
    case AGPU_VR_CONTROLLER_AXIS_JOYSTICK:
        joysticAxisState = Vector2(sourceState.x, sourceState.y);
        //printf("%d: joysticAxisState %f %f\n", index, joysticAxisState.x, joysticAxisState.y);
        break;
    case AGPU_VR_CONTROLLER_AXIS_TRIGGER:
        if(index == 1)
            triggerAxisState = Vector2(sourceState.x, sourceState.y);
        //printf("%d: triggerAxisState %f %f\n", index, triggerAxisState.x, triggerAxisState.y);
        break;
    default:
        break;
    }
}

int
Molevis::mainStart(int argc, const char *argv[])
{
    bool vsyncDisabled = false;
    bool debugLayerEnabled = false;
#ifdef _DEBUG
    debugLayerEnabled= true;
#endif
    agpu_uint platformIndex = 0;
    agpu_uint gpuIndex = 0;
    int randomAtomCount = 1000;
    int randomBondCount = 0;
    loadPeriodicTable();
    initializeAtomColorConventions();
    std::string inputFileName;

    for (int i = 1; i < argc; ++i)
    {
        std::string arg = argv[i];
        if(arg[0] == '-')
        {
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
            else if (arg == "-gen-atoms")
            {
                randomAtomCount = atoi(argv[++i]);
            }
            else if (arg == "-gen-bonds")
            {
                randomBondCount = atoi(argv[++i]);
            }
            else if (arg == "-paused")
            {
                isSimulating.store(false);
            }
            else if (arg == "-scale-factor")
            {
                modelScaleFactor = float(atof(argv[++i]));
            }                
            else if (arg == "-stereo")
            {
                isStereo = true;
            }
            else if (arg == "-vr")
            {
                isVirtualReality = true;
                vsyncDisabled = true;
            }
            
        }
        else
        {
            inputFileName = arg;
        }
    }

    if(!inputFileName.empty())
    {
        isSimulating.store(false);
        chemfiles::Trajectory file(inputFileName);
        chemfiles::Frame frame = file.read();

        convertChemfileFrame(frame);

    }
    else
    {
        generateTestDataset();
        //generateRandomDataset(randomAtomCount, randomBondCount);
    }

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

    window = SDL_CreateWindow("Mollevis", SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED, cameraState.screenWidth, cameraState.screenHeight, SDL_WINDOW_SHOWN | SDL_WINDOW_RESIZABLE | SDL_WINDOW_ALLOW_HIGHDPI);
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
    if(isVirtualReality)
        openInfo.open_flags = AGPU_DEVICE_OPEN_FLAG_ALLOW_VR;

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

    currentSwapChainCreateInfo.colorbuffer_format = swapChainColorBufferFormat;
    currentSwapChainCreateInfo.depth_stencil_format = AGPU_TEXTURE_FORMAT_UNKNOWN;
    currentSwapChainCreateInfo.width = cameraState.screenWidth;
    currentSwapChainCreateInfo.height = cameraState.screenHeight;
    currentSwapChainCreateInfo.buffer_count = 3;
    currentSwapChainCreateInfo.flags = AGPU_SWAP_CHAIN_FLAG_APPLY_SCALE_FACTOR_FOR_HI_DPI;
    if (vsyncDisabled || isVirtualReality)
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

    printf("Choosen device: %s\n", device->getName());

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

    if(isVirtualReality)
    {
        vrSystem = device->getVRSystem();
        if(vrSystem)
        {
            agpu_size2d targetSize;
            vrSystem->getRecommendedRenderTargetSize(&targetSize);
            vrDisplayWidth = targetSize.width;
            vrDisplayHeight = targetSize.height;
        }
        else
        {
            isVirtualReality = false;
        }
    }

    // Create the render pass
    {
        agpu_renderpass_color_attachment_description colorAttachment = {};
        colorAttachment.format = colorBufferFormat;
        colorAttachment.begin_action = AGPU_ATTACHMENT_CLEAR;
        colorAttachment.end_action = AGPU_ATTACHMENT_KEEP;
        colorAttachment.clear_value.r = 0.1f;
        colorAttachment.clear_value.g = 0.1f;
        colorAttachment.clear_value.b = 0.1f;
        colorAttachment.clear_value.a = 0.0f;
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

    // Create the output render pass
    {
        agpu_renderpass_color_attachment_description colorAttachment = {};
        colorAttachment.format = swapChainColorBufferFormat;
        colorAttachment.begin_action = AGPU_ATTACHMENT_CLEAR;
        colorAttachment.end_action = AGPU_ATTACHMENT_KEEP;
        colorAttachment.clear_value.r = 0.0f;
        colorAttachment.clear_value.g = 0.0f;
        colorAttachment.clear_value.b = 0.0f;
        colorAttachment.clear_value.a = 0.0f;
        colorAttachment.sample_count = 1;

        agpu_renderpass_description description = {};
        description.color_attachment_count = 1;
        description.color_attachments = &colorAttachment;

        outputRenderPass = device->createRenderPass(&description);
    }

    // Create the shader signature
    {

        auto builder = device->createShaderSignatureBuilder();
        // Sampler
        builder->beginBindingBank(1); // Set 0
        builder->addBindingBankElement(AGPU_SHADER_BINDING_TYPE_SAMPLER, 1);

        builder->beginBindingBank(2); // Set 1
        builder->addBindingBankElement(AGPU_SHADER_BINDING_TYPE_UNIFORM_BUFFER, 1); // Screen and UI state
        builder->addBindingBankElement(AGPU_SHADER_BINDING_TYPE_STORAGE_BUFFER, 1); // UI Data
        builder->addBindingBankElement(AGPU_SHADER_BINDING_TYPE_SAMPLED_IMAGE, 1); // Bitmap font
        builder->addBindingBankElement(AGPU_SHADER_BINDING_TYPE_SAMPLED_IMAGE, 1); // Hdr source
        builder->addBindingBankElement(AGPU_SHADER_BINDING_TYPE_SAMPLED_IMAGE, 1); // Left eye
        builder->addBindingBankElement(AGPU_SHADER_BINDING_TYPE_SAMPLED_IMAGE, 1); // Right eye

        builder->beginBindingBank(2); // Set 2
        builder->addBindingBankElement(AGPU_SHADER_BINDING_TYPE_STORAGE_BUFFER, 1); // Atom Description Buffer
        builder->addBindingBankElement(AGPU_SHADER_BINDING_TYPE_STORAGE_BUFFER, 1); // Atom Bond Buffer
        builder->addBindingBankElement(AGPU_SHADER_BINDING_TYPE_STORAGE_BUFFER, 1); // Atom Front State Buffer

        builder->beginBindingBank(2); // Set 3
        builder->addBindingBankElement(AGPU_SHADER_BINDING_TYPE_STORAGE_BUFFER, 1); // Bounding Quad Buffer

        builder->beginBindingBank(32); // Set 4
        builder->addBindingBankElement(AGPU_SHADER_BINDING_TYPE_UNIFORM_BUFFER, 1); // ModelState
        builder->addBindingBankElement(AGPU_SHADER_BINDING_TYPE_SAMPLED_IMAGE, 1); // Model texture

        builder->addBindingConstant(); // Highlighted atom

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
        leftEyeCameraStateUniformBuffer = device->createBuffer(&desc, nullptr);
    }
    {
        agpu_buffer_description desc = {};
        desc.size = (sizeof(CameraState) + 255) & (-256);
        desc.heap_type = AGPU_MEMORY_HEAP_TYPE_HOST_TO_DEVICE;
        desc.usage_modes = agpu_buffer_usage_mask(AGPU_COPY_DESTINATION_BUFFER | AGPU_UNIFORM_BUFFER);
        desc.main_usage_mode = AGPU_UNIFORM_BUFFER;
        desc.mapping_flags = AGPU_MAP_DYNAMIC_STORAGE_BIT;
        rightEyeCameraStateUniformBuffer = device->createBuffer(&desc, nullptr);
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
        bitmapFontInverseWidth = 1.0f / float(desc.width);
        bitmapFontInverseHeight = 1.0f / float(desc.height);
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
        atomDescriptionBuffer->uploadBufferData(0, agpu_size(sizeof(AtomDescription)*atomDescriptions.size()), atomDescriptions.data());
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
        atomBondDescriptionBuffer->uploadBufferData(0, agpu_size(sizeof(AtomBondDescription)*atomBondDescriptions.size()), atomBondDescriptions.data());
    }

    // Atom state buffers
    {
        agpu_buffer_description desc = {};
        desc.size = (sizeof(AtomState)*std::max(renderingAtomState.size(), size_t(1024)) + 255) & (-256);
        desc.heap_type = AGPU_MEMORY_HEAP_TYPE_DEVICE_LOCAL;
        desc.usage_modes = agpu_buffer_usage_mask(AGPU_COPY_DESTINATION_BUFFER | AGPU_STORAGE_BUFFER);
        desc.main_usage_mode = AGPU_STORAGE_BUFFER;
        desc.mapping_flags = AGPU_MAP_DYNAMIC_STORAGE_BIT;
        
        size_t uploadSize = sizeof(AtomState)*renderingAtomState.size();
        atomStateFrontBuffer = device->createBuffer(&desc, nullptr);
        atomStateFrontBuffer->uploadBufferData(0, agpu_size(uploadSize), renderingAtomState.data());
    }

    atomFrontBufferBinding = shaderSignature->createShaderResourceBinding(2);
    atomFrontBufferBinding->bindStorageBuffer(0, atomDescriptionBuffer);
    atomFrontBufferBinding->bindStorageBuffer(1, atomBondDescriptionBuffer);
    atomFrontBufferBinding->bindStorageBuffer(2, atomStateFrontBuffer);

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

    // Atom screen quad computation shader
    atomScreenQuadBufferComputationPipeline = compileAndBuildComputeShaderPipelineWithSourceFile("assets/shaders/shaderCommon.glsl", "assets/shaders/atomScreenQuadComputation.glsl");

    // Atom and bond draw pipeline state.
    screenBoundQuadVertex = compileShaderWithCommonSourceFile("assets/shaders/shaderCommon.glsl", "assets/shaders/screenBoundQuadVertex.glsl", AGPU_VERTEX_SHADER);
    atomDrawFragment = compileShaderWithCommonSourceFile("assets/shaders/shaderCommon.glsl", "assets/shaders/atomFragment.glsl", AGPU_FRAGMENT_SHADER);
    bondDrawVertex = compileShaderWithCommonSourceFile("assets/shaders/shaderCommon.glsl", "assets/shaders/bondVertex.glsl", AGPU_VERTEX_SHADER);
    bondDrawFragment = compileShaderWithCommonSourceFile("assets/shaders/shaderCommon.glsl", "assets/shaders/bondFragment.glsl", AGPU_FRAGMENT_SHADER);
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

    {
        auto builder = device->createPipelineBuilder();
        builder->setRenderTargetFormat(0, colorBufferFormat);
        builder->setDepthStencilFormat(depthBufferFormat);
        builder->setShaderSignature(shaderSignature);
        builder->attachShader(bondDrawVertex);
        builder->attachShader(bondDrawFragment);
        builder->setPrimitiveType(AGPU_TRIANGLE_STRIP);
        builder->setDepthState(false, false, AGPU_ALWAYS);
        bondXRayDrawPipeline = finishBuildingPipeline(builder);
    }

    cameraState.flipVertically = device->hasTopLeftNdcOrigin() == device->hasBottomLeftTextureCoordinates();

    // Floor grid
    {
        auto gridVertexShader = compileShaderWithCommonSourceFile("assets/shaders/shaderCommon.glsl", "assets/shaders/floorGridVertex.glsl", AGPU_VERTEX_SHADER);
        auto gridFragmentShader = compileShaderWithCommonSourceFile("assets/shaders/shaderCommon.glsl", "assets/shaders/floorGridFragment.glsl", AGPU_FRAGMENT_SHADER);
        auto builder = device->createPipelineBuilder();
        builder->setRenderTargetFormat(0, colorBufferFormat);
        builder->setDepthStencilFormat(depthBufferFormat);
        builder->setShaderSignature(shaderSignature);
        builder->attachShader(gridVertexShader);
        builder->attachShader(gridFragmentShader);
        builder->setPrimitiveType(AGPU_TRIANGLE_STRIP);
        builder->setDepthState(true, false, AGPU_GREATER_EQUAL);
        builder->setCullMode(AGPU_CULL_MODE_NONE);
        builder->setBlendState(-1, true);
        builder->setBlendFunction(-1, AGPU_BLENDING_ONE, AGPU_BLENDING_INVERTED_SRC_ALPHA, AGPU_BLENDING_OPERATION_ADD,
            AGPU_BLENDING_ONE, AGPU_BLENDING_INVERTED_SRC_ALPHA, AGPU_BLENDING_OPERATION_ADD);
        floorGridDrawPipeline = finishBuildingPipeline(builder);
    }

    // Model
    {
        agpu_size vertexStride = sizeof(agpu_vr_render_model_vertex);
        modelVertexLayout = device->createVertexLayout();
        modelVertexLayout->addVertexAttributeBindings(1, &vertexStride, VRRenderModelVertexDescSize, VRRenderModelVertexDesc);

        auto modelVertex = compileShaderWithCommonSourceFile("assets/shaders/shaderCommon.glsl", "assets/shaders/modelVertex.glsl", AGPU_VERTEX_SHADER);
        auto modelFragment = compileShaderWithCommonSourceFile("assets/shaders/shaderCommon.glsl", "assets/shaders/modelFragment.glsl", AGPU_FRAGMENT_SHADER);

        auto builder = device->createPipelineBuilder();
        builder->setRenderTargetFormat(0, colorBufferFormat);
        builder->setDepthStencilFormat(depthBufferFormat);
        builder->setShaderSignature(shaderSignature);
        builder->attachShader(modelVertex);
        builder->attachShader(modelFragment);
        builder->setPrimitiveType(AGPU_TRIANGLES);
        builder->setDepthState(true, true, AGPU_GREATER_EQUAL);
        builder->setVertexLayout(modelVertexLayout);
        builder->setCullMode(AGPU_CULL_MODE_NONE);
        modelPipelineState = finishBuildingPipeline(builder);
    }

    // Pointer
    {
        agpu_size vertexStride = sizeof(PackedVector3);
        pointerModelVertexLayout = device->createVertexLayout();
        pointerModelVertexLayout->addVertexAttributeBindings(1, &vertexStride, VRPointerModelVertexDescSize, VRPointerModelVertexDesc);

        auto pointerModelVertex = compileShaderWithCommonSourceFile("assets/shaders/shaderCommon.glsl", "assets/shaders/pointerModelVertex.glsl", AGPU_VERTEX_SHADER);
        auto pointerModelFragment = compileShaderWithCommonSourceFile("assets/shaders/shaderCommon.glsl", "assets/shaders/pointerModelFragment.glsl", AGPU_FRAGMENT_SHADER);

        auto builder = device->createPipelineBuilder();
        builder->setRenderTargetFormat(0, colorBufferFormat);
        builder->setDepthStencilFormat(depthBufferFormat);
        builder->setShaderSignature(shaderSignature);
        builder->attachShader(pointerModelVertex);
        builder->attachShader(pointerModelFragment);
        builder->setPrimitiveType(AGPU_TRIANGLES);
        builder->setDepthState(true, true, AGPU_GREATER_EQUAL);
        builder->setVertexLayout(pointerModelVertexLayout);
        builder->setCullMode(AGPU_CULL_MODE_NONE);
        builder->setBlendFunction(-1,
            AGPU_BLENDING_ONE, AGPU_BLENDING_INVERTED_SRC_ALPHA, AGPU_BLENDING_OPERATION_ADD,
            AGPU_BLENDING_ONE, AGPU_BLENDING_INVERTED_SRC_ALPHA, AGPU_BLENDING_OPERATION_ADD
        );
        builder->setBlendState(-1, true);
        builder->setPrimitiveType(AGPU_TRIANGLES);
        pointerModelPipelineState = finishBuildingPipeline(builder);
    }

    // Tonemapping
    if(cameraState.flipVertically)
        screenQuadVertex = compileShaderWithSourceFile("assets/shaders/screenQuadFlipped.glsl", AGPU_VERTEX_SHADER);
    else
        screenQuadVertex = compileShaderWithSourceFile("assets/shaders/screenQuad.glsl", AGPU_VERTEX_SHADER);
    filmicTonemappingFragment = compileShaderWithSourceFile("assets/shaders/filmicTonemapping.glsl", AGPU_FRAGMENT_SHADER);
    {
        auto builder = device->createPipelineBuilder();
        builder->setRenderTargetFormat(0, swapChainColorBufferFormat);
        builder->setDepthStencilFormat(AGPU_TEXTURE_FORMAT_UNKNOWN);
        builder->setShaderSignature(shaderSignature);
        builder->attachShader(screenQuadVertex);
        builder->attachShader(filmicTonemappingFragment);
        builder->setPrimitiveType(AGPU_TRIANGLE_STRIP);
        filmicTonemappingPipeline = finishBuildingPipeline(builder);
    }

    passthroughFragment  = compileShaderWithSourceFile("assets/shaders/passthroughLeft.glsl", AGPU_FRAGMENT_SHADER);
    {
        auto builder = device->createPipelineBuilder();
        builder->setRenderTargetFormat(0, swapChainColorBufferFormat);
        builder->setDepthStencilFormat(AGPU_TEXTURE_FORMAT_UNKNOWN);
        builder->setShaderSignature(shaderSignature);
        builder->attachShader(screenQuadVertex);
        builder->attachShader(passthroughFragment);
        builder->setPrimitiveType(AGPU_TRIANGLE_STRIP);
        passthroughPipeline = finishBuildingPipeline(builder);
    }

    sideBySideFragment  = compileShaderWithSourceFile("assets/shaders/sideBySide.glsl", AGPU_FRAGMENT_SHADER);
    {
        auto builder = device->createPipelineBuilder();
        builder->setRenderTargetFormat(0, swapChainColorBufferFormat);
        builder->setDepthStencilFormat(AGPU_TEXTURE_FORMAT_UNKNOWN);
        builder->setShaderSignature(shaderSignature);
        builder->attachShader(screenQuadVertex);
        builder->attachShader(sideBySideFragment);
        builder->setPrimitiveType(AGPU_TRIANGLE_STRIP);
        sideBySidePipeline = finishBuildingPipeline(builder);
    }
    

    // Data binding
    leftEyeScreenStateBinding = shaderSignature->createShaderResourceBinding(1);
    leftEyeScreenStateBinding->bindUniformBuffer(0, leftEyeCameraStateUniformBuffer);
    leftEyeScreenStateBinding->bindStorageBuffer(1, uiDataBuffer);
    leftEyeScreenStateBinding->bindSampledTextureView(2, bitmapFont->getOrCreateFullView());

    rightEyeScreenStateBinding = shaderSignature->createShaderResourceBinding(1);
    rightEyeScreenStateBinding->bindUniformBuffer(0, rightEyeCameraStateUniformBuffer);
    rightEyeScreenStateBinding->bindStorageBuffer(1, uiDataBuffer);
    rightEyeScreenStateBinding->bindSampledTextureView(2, bitmapFont->getOrCreateFullView());

    // UI pipeline state.
    uiElementVertex = compileShaderWithCommonSourceFile("assets/shaders/shaderCommon.glsl", "assets/shaders/uiElementVertex.glsl", AGPU_VERTEX_SHADER);
    uiElementFragment = compileShaderWithSourceFile("assets/shaders/uiElementFragment.glsl", AGPU_FRAGMENT_SHADER);

    createPointerModel();
    createIntermediateTexturesAndFramebuffer();

    if(!uiElementVertex || !uiElementFragment)
        return 1;

    {
        auto builder = device->createPipelineBuilder();
        builder->setRenderTargetFormat(0, swapChainColorBufferFormat);
        builder->setDepthStencilFormat(AGPU_TEXTURE_FORMAT_UNKNOWN);
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

    // Start the simulation thread
    startSimulationThread();

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

    {
        std::unique_lock l(simulationThreadStateMutex);
        isSimulating.store(false);
        simulationThreadStateConditionChanged.notify_all();
    }
    simulationThread.join();

    commandQueue->finishExecution();
    swapChain.reset();
    commandQueue.reset();

    SDL_DestroyWindow(window);
    SDL_Quit();
    return 0;
}

void
Molevis::createIntermediateTexturesAndFramebuffer()
{
    if(isVirtualReality)
    {
        if(framebufferDisplayWidth == vrDisplayWidth && framebufferDisplayHeight == vrDisplayHeight)
            return;

        framebufferDisplayWidth = vrDisplayWidth;
        framebufferDisplayHeight = vrDisplayHeight;
        printf("Make intermediate %d %d\n", framebufferDisplayWidth, framebufferDisplayHeight);   
    }
    else
    {
        if(framebufferDisplayWidth == displayWidth && framebufferDisplayHeight == displayHeight)
            return;

        framebufferDisplayWidth = displayWidth;
        framebufferDisplayHeight = displayHeight;        
    }


    // Depth stencil
    {
        agpu_texture_description desc = {};
        desc.type = AGPU_TEXTURE_2D;
        desc.width = framebufferDisplayWidth;
        desc.height = framebufferDisplayHeight;
        desc.depth = 1;
        desc.layers = 1;
        desc.miplevels = 1;
        desc.format = depthBufferFormat;
        desc.usage_modes = AGPU_TEXTURE_USAGE_DEPTH_ATTACHMENT;
        desc.main_usage_mode = AGPU_TEXTURE_USAGE_DEPTH_ATTACHMENT;
        desc.heap_type = AGPU_MEMORY_HEAP_TYPE_DEVICE_LOCAL;
        desc.sample_count = 1;
        desc.sample_quality = 0;

        depthStencilTexture = device->createTexture(&desc);
    }

    // Color targets
    {
        agpu_texture_description desc = {};
        desc.type = AGPU_TEXTURE_2D;
        desc.width = framebufferDisplayWidth;
        desc.height = framebufferDisplayHeight;
        desc.depth = 1;
        desc.layers = 1;
        desc.miplevels = 1;
        desc.format = colorBufferFormat;
        desc.usage_modes = agpu_texture_usage_mode_mask(AGPU_TEXTURE_USAGE_COLOR_ATTACHMENT | AGPU_TEXTURE_USAGE_SAMPLED | AGPU_TEXTURE_USAGE_COPY_SOURCE);
        desc.main_usage_mode = AGPU_TEXTURE_USAGE_SAMPLED;
        desc.heap_type = AGPU_MEMORY_HEAP_TYPE_DEVICE_LOCAL;
        desc.sample_count = 1;
        desc.sample_quality = 0;

        hdrTargetTexture = device->createTexture(&desc);
    }

    // Color output targets
    {
        agpu_texture_description desc = {};
        desc.type = AGPU_TEXTURE_2D;
        desc.width = framebufferDisplayWidth;
        desc.height = framebufferDisplayHeight;
        desc.depth = 1;
        desc.layers = 1;
        desc.miplevels = 1;
        desc.format = swapChainColorBufferFormat;
        desc.usage_modes = agpu_texture_usage_mode_mask(AGPU_TEXTURE_USAGE_COLOR_ATTACHMENT | AGPU_TEXTURE_USAGE_SAMPLED | AGPU_TEXTURE_USAGE_COPY_SOURCE);
        desc.main_usage_mode = AGPU_TEXTURE_USAGE_SAMPLED;
        desc.heap_type = AGPU_MEMORY_HEAP_TYPE_DEVICE_LOCAL;
        desc.sample_count = 1;
        desc.sample_quality = 0;

        leftEyeTexture = device->createTexture(&desc);
        rightEyeTexture = device->createTexture(&desc);
    }

    auto depthStencilView = depthStencilTexture->getOrCreateFullView();
    auto hdrTargetTextureView = hdrTargetTexture->getOrCreateFullView();
    hdrTargetFramebuffer = device->createFrameBuffer(framebufferDisplayWidth, framebufferDisplayHeight, 1, &hdrTargetTextureView, depthStencilView);

    auto leftEyeView = leftEyeTexture->getOrCreateFullView();
    leftEyeFramebuffer = device->createFrameBuffer(framebufferDisplayWidth, framebufferDisplayHeight, 1, &leftEyeView, nullptr);

    auto rightEyeView = rightEyeTexture->getOrCreateFullView();
    rightEyeFramebuffer = device->createFrameBuffer(framebufferDisplayWidth, framebufferDisplayHeight, 1, &rightEyeView, nullptr);

    leftEyeScreenStateBinding->bindSampledTextureView(3, hdrTargetTextureView);
    leftEyeScreenStateBinding->bindSampledTextureView(4, leftEyeView);
    leftEyeScreenStateBinding->bindSampledTextureView(5, rightEyeView);

    rightEyeScreenStateBinding->bindSampledTextureView(3, hdrTargetTextureView);
    rightEyeScreenStateBinding->bindSampledTextureView(4, leftEyeView);
    rightEyeScreenStateBinding->bindSampledTextureView(5, rightEyeView);
}

void
Molevis::createPointerModel()
{
    AABox box;
    box.min = Vector3(-0.01f, -0.01f, -1000.1f);
    box.max = Vector3(0.01f, 0.01f, 0.01f);

    auto minX = box.min.x;
    auto minY = box.min.y;
    auto minZ = box.min.z;
    auto maxX = box.max.x;
    auto maxY = box.max.y;
    auto maxZ = box.max.z;

    PackedVector3 vertices[] = {
        // Left
		{minX, minY, minZ},
		{minX, maxY, minZ},
		{minX, maxY, maxZ},
		{minX, minY, maxZ},

        // Right
		{maxX, minY, minZ},
		{maxX, maxY, minZ},
		{maxX, maxY, maxZ},
		{maxX, minY, maxZ},

        // Top
        {minX, maxY, minZ},
		{maxX, maxY, minZ},
		{maxX, maxY, maxZ},
		{minX, maxY, maxZ},

        // Bottom
        {minX, minY, minZ},
        {maxX, minY, minZ},
        {maxX, minY, maxZ},
        {minX, minY, maxZ},

        // Back
        {minX, minY, minZ},
		{maxX, minY, minZ},
		{maxX, maxY, minZ},
		{minX, maxY, minZ},
        
        // Back
        {minX, minY, maxZ},
		{maxX, minY, maxZ},
		{maxX, maxY, maxZ},
		{minX, maxY, maxZ},
    };
    uint16_t indices[] = {
        // Left
		0 + 1, 0 + 0, 0 + 2,
		0 + 3, 0 + 2, 0 + 0,

        // Right
		4 + 0, 4 + 1, 4 + 2,
		4 + 2, 4 + 3, 4 + 0,

        // Top
		8 + 1, 8 + 0, 8 + 2,
		8 + 3, 8 + 2, 8 + 0,

        // Bottom
		12 + 0, 12 + 1, 12 + 2,
		12 + 2, 12 + 3, 12 + 0,

        // Back
        16 + 1, 16 + 0, 16 + 2,
		16 + 3, 16 + 2, 16 + 0,

        // Front
        20 + 0, 20 + 1, 20 + 2,
		20 + 2, 20 + 3, 20 + 0,
    };

    pointerModel = std::make_shared<TrackedDeviceModel> ();
    auto vertexBufferSize = sizeof(vertices);
    auto indexBufferSize = sizeof(indices);

    {
        agpu_buffer_description desc = {0};
        desc.size = agpu_size(vertexBufferSize);
        desc.heap_type = AGPU_MEMORY_HEAP_TYPE_DEVICE_LOCAL;
        desc.usage_modes = agpu_buffer_usage_mask(AGPU_COPY_DESTINATION_BUFFER | AGPU_ARRAY_BUFFER);
        desc.main_usage_mode = AGPU_ARRAY_BUFFER;
        desc.mapping_flags = AGPU_MAP_DYNAMIC_STORAGE_BIT;

        pointerModel->vertexBuffer = device->createBuffer(&desc, vertices);
        pointerModel->vertexCount = sizeof(vertices) / sizeof(vertices[0]);
    }
    {
        pointerModel->vertexBinding = device->createVertexBinding(pointerModelVertexLayout);
        pointerModel->vertexBinding->bindVertexBuffers(1, &pointerModel->vertexBuffer);
    }

    {
        agpu_buffer_description desc = {0};
        desc.size = agpu_size(indexBufferSize);
        desc.heap_type = AGPU_MEMORY_HEAP_TYPE_DEVICE_LOCAL;
        desc.usage_modes = agpu_buffer_usage_mask(AGPU_COPY_DESTINATION_BUFFER | AGPU_ELEMENT_ARRAY_BUFFER);
        desc.main_usage_mode = AGPU_ELEMENT_ARRAY_BUFFER;
        desc.mapping_flags = AGPU_MAP_DYNAMIC_STORAGE_BIT;
        desc.stride = 2;

        pointerModel->indexBuffer = device->createBuffer(&desc, indices);
        pointerModel->indexCount = sizeof(indices) / sizeof(indices[0]);
    }
}

void
Molevis::initializeAtomColorConventions()
{
    atomTypeColorMap["H"] = Vector4(0.4f, 0.4f, 0.4f, 1.0f);
    atomTypeColorMap["C"] = Vector4(0.1f, 0.1f, 0.1f, 1.0f);
    atomTypeColorMap["N"] = Vector4(0.1f, 0.1f, 0.8f, 1.0f);
    atomTypeColorMap["O"] = Vector4(0.8f, 0.1f, 0.1f, 1.0f);
}

void
Molevis::loadPeriodicTable()
{
    if(!periodicTable.loadFromFile("assets/datasets/periodic-table-of-elements.csv"))
    {
        fprintf(stderr, "Failed to load periodic table dataset.\n");
        abort();
    }

    if(!LennardJonesTable().loadFromFile("assets/datasets/lennard-jones-table.csv", periodicTable))
    {
        fprintf(stderr, "Failed to load lennard-jones coefficient table.\n");
        abort();
    }

}

Vector4
Molevis::getOrCreateColorForAtomType(const std::string &type)
{
    auto it = atomTypeColorMap.find(type);
    if(it != atomTypeColorMap.end())
        return it->second;
    
    auto generatedColor = randColor.randVector4(Vector4{0.1f, 0.1f, 0.1f, 1.0f}, Vector4{0.8f, 0.8f, 0.8f, 1.0f});
    atomTypeColorMap.insert(std::make_pair(type, generatedColor));
    return generatedColor;
}

void
Molevis::convertChemfileFrame(chemfiles::Frame &frame)
{
    Random rand;
    const auto &positions = frame.positions();
    atomDescriptions.reserve(frame.size());
    simulationAtomState.reserve(frame.size());
    renderingAtomState.reserve(frame.size());
    
    for(size_t i = 0; i < positions.size(); ++i)
    {
        const auto &atomPosition = positions[i];
        const auto &chemAtom = frame[i];

        auto chemAtomNumber = chemAtom.atomic_number();

        auto description = AtomDescription{};
        description.atomNumber = chemAtomNumber ? int(chemAtomNumber.value()) : 1;
        description.mass = float(chemAtom.mass());
        
        description.color = getOrCreateColorForAtomType(chemAtom.type());
        description.radius = 0.2f;
        auto covalentRadius = chemAtom.covalent_radius();
        if(covalentRadius)
            description.radius = float(covalentRadius.value());

        auto &periodicElement = periodicTable.elements[description.atomNumber];
        {
            description.lennardJonesCutoff = float(periodicElement.lennardJonesCutoff);
            description.lennardJonesEpsilon = float(periodicElement.lennardJonesEpsilon);
            description.lennardJonesSigma = float(periodicElement.lennardJonesSigma);
        }

        {
            auto simulationState = AtomSimulationState{};
            simulationState.position = DVector3(atomPosition[0], atomPosition[1], atomPosition[2]);

            atomDescriptions.push_back(description);
            simulationAtomState.push_back(simulationState);
            renderingAtomState.push_back(simulationState.asRenderingState());
        }
    }

    auto& topology = frame.topology();
    for(auto &bond : topology.bonds())
    {
        auto firstAtomIndex = bond[0];
        auto secondAtomIndex = bond[1];

        const auto &firstAtomDesc = atomDescriptions[firstAtomIndex];
        const auto &secondAtomDesc = atomDescriptions[secondAtomIndex];

        const auto &firstAtomPosition = renderingAtomState[firstAtomIndex];
        const auto &secondAtomPosition = renderingAtomState[secondAtomIndex];

        auto atomEquilibriumDistance = (firstAtomPosition.position - secondAtomPosition.position).length();
        //auto atomEquilibriumDistance = firstAtomDesc.radius + secondAtomDesc.radius;

        auto description = AtomBondDescription{};
        description.firstAtomIndex = uint32_t(firstAtomIndex);
        description.secondAtomIndex = uint32_t(secondAtomIndex);
        
        description.equilibriumDistance = atomEquilibriumDistance;
        description.morseWellDepth = 1;
        description.morseWellWidth = 1;
        description.thickness = 0.1f;
        description.color = Vector4{0.8f, 0.8f, 0.8f, 1.0f};
        atomBondDescriptions.push_back(description);
    }

    computeAtomsBoundingBox();
}

void
Molevis::generateTestDataset()
{
    atomDescriptions.reserve(3);
    renderingAtomState.reserve(3);
    simulationAtomState.reserve(3);

    auto hydrogenDesc = periodicTable.makeAtomDescriptionForSymbol("H");
    hydrogenDesc.color = getOrCreateColorForAtomType("H");

    auto oxygenDesc = periodicTable.makeAtomDescriptionForSymbol("O");
    oxygenDesc.color = getOrCreateColorForAtomType("O");
    //atomDescriptions.push_back(description);
    
    {
        auto state = AtomSimulationState{};
        state.position = DVector3(-5, 0.0, 0.0);
        atomDescriptions.push_back(hydrogenDesc);
        renderingAtomState.push_back(state.asRenderingState());
        simulationAtomState.push_back(state);
    }

    {
        auto state = AtomSimulationState{};
        state.position = DVector3(5, 0.0, 0.0);
        atomDescriptions.push_back(hydrogenDesc);
        renderingAtomState.push_back(state.asRenderingState());
        simulationAtomState.push_back(state);
    }

    {
        auto state = AtomSimulationState{};
        state.position = DVector3(0, 0.0, 0.0);
        atomDescriptions.push_back(oxygenDesc);
        renderingAtomState.push_back(state.asRenderingState());
        simulationAtomState.push_back(state);
    }

    {
        AtomBondDescription bond = {};
        bond.color = Vector4(0.8f, 0.8f, 0.8f, 1.0f);
        bond.firstAtomIndex = 0;
        bond.secondAtomIndex = 2;
        bond.thickness = 0.1f;
        bond.equilibriumDistance = hydrogenDesc.radius + oxygenDesc.radius;
        bond.morseWellDepth = 1;
        bond.morseWellWidth = 1;
        atomBondDescriptions.push_back(bond);
    }

    {
        AtomBondDescription bond = {};
        bond.color = Vector4(0.8f, 0.8f, 0.8f, 1.0f);
        bond.firstAtomIndex = 1;
        bond.secondAtomIndex = 2;
        bond.thickness = 0.1f;
        bond.equilibriumDistance = hydrogenDesc.radius + oxygenDesc.radius;
        bond.morseWellDepth = 1;
        bond.morseWellWidth = 1;
        atomBondDescriptions.push_back(bond);
    }
}

void
Molevis::generateRandomDataset(size_t atomsToGenerate, size_t bondsToGenerate)
{
    Random rand;
    atomDescriptions.reserve(atomsToGenerate);
    renderingAtomState.reserve(atomsToGenerate);
    simulationAtomState.reserve(atomsToGenerate);

    for(size_t i = 0; i < atomsToGenerate; ++i)
    {
        auto description = AtomDescription{};
        auto state = AtomSimulationState{};

        description.lennardJonesEpsilon = 1.0f;//rand.randFloat(1, 5);
        description.lennardJonesSigma = 1.0f;//rand.randFloat(1, 5);
        description.radius = 1.0f;//rand.randFloat(0.5, 2);
        description.color = rand.randVector4(Vector4{0.1f, 0.1f, 0.1f, 1.0f}, Vector4{0.8f, 0.8f, 0.8f, 1.0f});
        description.mass = 1.0f;
        state.position = rand.randDVector3(-10, 10);

        atomDescriptions.push_back(description);
        simulationAtomState.push_back(state);
        renderingAtomState.push_back(state.asRenderingState());
    }

    for(size_t i = 0; i < bondsToGenerate; ++i)
    {
        auto firstAtomIndex = rand.randUInt(uint32_t(atomDescriptions.size()));
        auto secondAtomIndex = firstAtomIndex;
        while(firstAtomIndex == secondAtomIndex)
            secondAtomIndex = rand.randUInt(uint32_t(atomDescriptions.size()));

        auto description = AtomBondDescription{};
        description.firstAtomIndex = firstAtomIndex;
        description.secondAtomIndex = secondAtomIndex;
        description.equilibriumDistance = rand.randFloat(5, 20);
        description.morseWellDepth = 1;
        description.morseWellWidth = 1;
        description.thickness = rand.randFloat(0.1f, 0.4f);
        description.color = rand.randVector4(Vector4{0.1f, 0.1f, 0.1f, 1.0}, Vector4{0.8f, 0.8f, 0.8f, 1.0f});
        atomBondDescriptions.push_back(description);
    }
    
    computeAtomsBoundingBox();
}

void
Molevis::computeAtomsBoundingBox()
{
    // Move the atoms to their bounding box center
    {
        atomsBoundingBox = DAABox::empty();
        for(auto &atom : simulationAtomState)
            atomsBoundingBox.insertPoint(atom.position);
        auto center = atomsBoundingBox.center();
        for(auto &atom : simulationAtomState)
            atom.position = atom.position - center;
    }

    // Recompute the bounding
    {
        atomsBoundingBox = DAABox::empty();
        for(auto &atom : simulationAtomState)
            atomsBoundingBox.insertPoint(atom.position);

        //modelPosition = Vector3(0.0, atomsBoundingBox.halfExtent().y*modelScaleFactor, 0.0);
        modelPosition = Vector3(0, 0, 0);
    }


    for(size_t i = 0;i < simulationAtomState.size(); ++i)
        renderingAtomState[i] = simulationAtomState[i].asRenderingState();

}

std::string
Molevis::readWholeFile(const std::string &fileName)
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

agpu_shader_ref
Molevis::compileShaderWithSourceFile(const std::string &sourceFileName, agpu_shader_type type)
{
    return compileShaderWithSource(sourceFileName, readWholeFile(sourceFileName), type);
}

agpu_shader_ref
Molevis::compileShaderWithCommonSourceFile(const std::string &commonSourceFile, const std::string &sourceFileName, agpu_shader_type type)
{
    return compileShaderWithSource(sourceFileName, readWholeFile(commonSourceFile) + readWholeFile(sourceFileName), type);
}

agpu_shader_ref
Molevis::compileShaderWithSource(const std::string &name, const std::string &source, agpu_shader_type type)
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

agpu_pipeline_state_ref
Molevis::finishBuildingPipeline(const agpu_pipeline_builder_ref &builder)
{
    auto pipeline = builder->build();
    if(!pipeline)
    {
        fprintf(stderr, "Failed to build pipeline.\n");
    }
    return pipeline;
}

agpu_pipeline_state_ref
Molevis::compileAndBuildComputeShaderPipelineWithSourceFile(const std::string &commonSourceFilename, const std::string &filename)
{
    auto shader = compileShaderWithCommonSourceFile(commonSourceFilename, filename, AGPU_COMPUTE_SHADER);
    auto builder = device->createComputePipelineBuilder();
    builder->setShaderSignature(shaderSignature);
    builder->attachShader(shader);
    return finishBuildingComputePipeline(builder);
}

agpu_pipeline_state_ref
Molevis::finishBuildingComputePipeline(const agpu_compute_pipeline_builder_ref &builder)
{
    auto pipeline = builder->build();
    if(!pipeline)
    {
        fprintf(stderr, "Failed to build pipeline.\n");
    }
    return pipeline;
}

void
Molevis::processEvents()
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

void
Molevis::processEvent(const SDL_Event &event)
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

void
Molevis::recreateSwapChain()
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
    createIntermediateTexturesAndFramebuffer();
}

void
Molevis::onKeyDown(const SDL_KeyboardEvent &event)
{
    switch(event.keysym.sym)
    {
    case SDLK_ESCAPE:
        isQuitting = true;
        break;
    case SDLK_KP_PLUS:
        modelScaleFactor *= 1.1f;
        break;
    case SDLK_KP_MINUS:
        modelScaleFactor /= 1.1f;
        break;
    case ' ':
    {
        isSimulating.store(!isSimulating.load());
        simulationThreadStateConditionChanged.notify_all();
    }
        break;
    case 'x':
        bondXRay = !bondXRay;
        break;
    default:
        break;
    }
}


void 
Molevis::onMouseMotion(const SDL_MouseMotionEvent &event)
{
    bool hasCtrlDown = SDL_GetModState() & KMOD_CTRL;

    if((event.state & SDL_BUTTON_LMASK) && !hasCtrlDown)
    {
        hasLeftDragEvent = true;
        leftDragStartX = event.x;
        leftDragStartY = event.y;
        leftDragDeltaX = event.xrel;
        leftDragDeltaY = event.yrel;
    }

    if((event.state & SDL_BUTTON_RMASK) || ((event.state & SDL_BUTTON_LMASK) && hasCtrlDown))
    {
        hasRightDragEvent = true;
        rightDragStartX = event.x;
        rightDragStartY = event.y;
        rightDragDeltaX = event.xrel;
        rightDragDeltaY = event.yrel;
    }
    
    mousePositionX = event.x;
    mousePositionY = event.y;
}

void Molevis::onMouseWheel(const SDL_MouseWheelEvent &event)
{
    hasWheelEvent = true;
    wheelDelta = event.y;
}

void Molevis::drawRectangle(const Vector2 &position, const Vector2 &size, const Vector4 &color)
{
    UIElementQuad quad = {};
    quad.position = position;
    quad.size = size;
    quad.color = color;

    uiElementQuadBuffer.push_back(quad);
}

Vector2 Molevis::drawGlyph(char c, const Vector2 &position, const Vector4 &color)
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

Vector2
Molevis::drawString(const std::string &string, const Vector2 &position, const Vector4 &color)
{
    Vector2 totalAdvance = {0, 0};
    for(auto c : string)
        totalAdvance += drawGlyph(c, position + totalAdvance, color);
    return totalAdvance;
}

void
Molevis::beginLayout(float x, float y)
{
    currentLayoutX = currentLayoutRowX = x;
    currentLayoutY = currentLayoutRowY = y;
}

void 
Molevis::advanceLayoutRow()
{
    currentLayoutRowY += bitmapFontGlyphHeight * bitmapFontScale + 5;
    currentLayoutX = currentLayoutRowX;
    currentLayoutY = currentLayoutRowY;
}

void
Molevis::simulateIterationInCPU(double timestep)
{
    assert(simulationAtomState.size() == renderingAtomState.size());
    // Reset the net force.
    for(size_t i = 0; i < simulationAtomState.size(); ++i)
        simulationAtomState[i].netForce = DVector3(0, 0, 0);

    // Lennard-jones potential
    for(size_t i = 0; i < simulationAtomState.size(); ++i)
    {
        auto &firstAtomDesc = atomDescriptions[i];
        auto &firstAtomState = simulationAtomState[i];

        DVector3 firstPosition = firstAtomState.position;

        double firstLennardJonesCutoff  = firstAtomDesc.lennardJonesCutoff;
        double firstLennardJonesEpsilon = firstAtomDesc.lennardJonesEpsilon;
        double firstLennardJonesSigma   = firstAtomDesc.lennardJonesSigma;

        for(size_t j = 0; j < simulationAtomState.size(); ++j)
        {
            if(i == j)
                continue;

            auto &secondAtomDesc = atomDescriptions[j];
            auto &secondAtomState = simulationAtomState[j];

            DVector3 secondPosition = secondAtomState.position;

            double secondLennardJonesCutoff  = secondAtomDesc.lennardJonesCutoff;
            double secondLennardJonesEpsilon = secondAtomDesc.lennardJonesEpsilon;
            double secondLennardJonesSigma   = secondAtomDesc.lennardJonesSigma;

            double lennardJonesCutoff = firstLennardJonesCutoff + secondLennardJonesCutoff;
            double lennardJonesEpsilon = sqrt(firstLennardJonesEpsilon*secondLennardJonesEpsilon);
            double lennardJonesSigma = (firstLennardJonesSigma + secondLennardJonesSigma) * 0.5;

            DVector3 direction = firstPosition - secondPosition;
            auto dist = direction.length();
            if(1e-6 < dist && dist < lennardJonesCutoff)
            {
                auto normalizedDirection = direction / dist;
                auto force = -normalizedDirection * lennardJonesDerivative(std::max(dist, 1.0), lennardJonesSigma, lennardJonesEpsilon);
                firstAtomState.netForce = firstAtomState.netForce + force;

                //if(i == 94 && j == 95)
                //    printf("Dist %zu %zu: %f\n", i, j, dist);
            }
        }
    }

    // Morse bond
    /*for(auto &bond : atomBondDescriptions)
    {
        auto &firstAtomState = simulationAtomState[bond.firstAtomIndex];
        auto &secondAtomState = simulationAtomState[bond.secondAtomIndex];

        auto direction = firstAtomState.position - secondAtomState.position;
        auto distance = direction.length();
        auto normalizedDirection = direction / distance;
        auto force = -normalizedDirection*morsePotentialDerivative(distance, bond.morseWellDepth, bond.morseWellWidth, bond.equilibriumDistance);
        firstAtomState.netForce = firstAtomState.netForce + force;
        secondAtomState.netForce = secondAtomState.netForce - force;
    }*/

    // Hooke law bond
    for(auto &bond : atomBondDescriptions)
    {
        auto &firstAtomState = simulationAtomState[bond.firstAtomIndex];
        auto &secondAtomState = simulationAtomState[bond.secondAtomIndex];

        auto direction = firstAtomState.position - secondAtomState.position;
        auto distance = direction.length();
        auto normalizedDirection = direction / distance;
        auto force = -normalizedDirection*hookPotentialDerivative(distance, bond.equilibriumDistance, 100.0);
        firstAtomState.netForce = firstAtomState.netForce + force;
        secondAtomState.netForce = secondAtomState.netForce - force;
    }

    // Integrate the velocites and compute total kinetic energy.
    double totalKineticEnergy = 0.0;
    for(size_t i = 0; i < simulationAtomState.size(); ++i)
    {
        auto &state = simulationAtomState[i];
        auto mass = atomDescriptions[i].mass;
        auto acceleration = state.netForce / mass;
        //printf("%zu %f %f %f\n", i, state.netForce.x, state.netForce.y, state.netForce.z);
        state.velocity = state.velocity + acceleration*timestep;
        totalKineticEnergy = totalKineticEnergy + 0.5*mass*state.velocity.length2();
    }

    // Compute the average kinetic energy.
    double averageKineticEnergy = totalKineticEnergy / double(simulationAtomState.size());
    //printf("total kinetic %f average %f\n", totalKineticEnergy, averageKineticEnergy);;

    double targetKineticEnergy = 1.0;
    double kineticEnergyLambda = targetKineticEnergy / std::max(0.01, averageKineticEnergy);
    //double kineticEnergyLambda = 1.0;
    //printf("lambda %f\n", kineticEnergyLambda);
    
    for(size_t i = 0; i < simulationAtomState.size(); ++i)
    {
        auto &state = simulationAtomState[i];
        state.velocity = state.velocity * kineticEnergyLambda;
    }
        


    // Integrate the positions
    for(size_t i = 0; i < simulationAtomState.size(); ++i)
    {
        auto &state = simulationAtomState[i];
        state.position = state.position + state.velocity*timestep;
    }


    // Upload the new state
    {
        std::unique_lock l(renderingAtomStateMutex);
        for(size_t i = 0; i < simulationAtomState.size(); ++i)
            renderingAtomState[i] = simulationAtomState[i].asRenderingState();
        renderingAtomStateDirty = true;
    }

    simulationIteration.fetch_add(1);
}

void
Molevis::simulationThreadEntry()
{
    while(!isQuitting.load())
    {
        bool shouldStepSimulation = false;
        {
            std::unique_lock l(simulationThreadStateMutex);
            while(!isQuitting && !isSimulating)
                simulationThreadStateConditionChanged.wait(l);

            if(isQuitting)
                return;
            shouldStepSimulation = isSimulating;
        }

        if(shouldStepSimulation)
            simulateIterationInCPU(SimulationTimeStep);
    }

}

void Molevis::startSimulationThread()
{
    std::thread t([=](){
        simulationThreadEntry();
    });
    simulationThread.swap(t);
}

TrackedDeviceModelPtr Molevis::loadDeviceModel(agpu_vr_render_model *agpuModel)
{
    auto model = std::make_shared<TrackedDeviceModel> ();
    model->vertexCount = agpuModel->vertex_count;
    model->indexCount = agpuModel->triangle_count*3;
    auto vertexBufferSize = agpuModel->vertex_count * sizeof(agpu_vr_render_model_vertex);
    auto indexBufferSize = model->indexCount*sizeof(uint16_t);

    {
        agpu_buffer_description desc = {0};
        desc.size = agpu_size(vertexBufferSize);
        desc.heap_type = AGPU_MEMORY_HEAP_TYPE_DEVICE_LOCAL;
        desc.usage_modes = agpu_buffer_usage_mask(AGPU_COPY_DESTINATION_BUFFER | AGPU_ARRAY_BUFFER);
        desc.main_usage_mode = AGPU_ARRAY_BUFFER;
        desc.mapping_flags = AGPU_MAP_DYNAMIC_STORAGE_BIT;

        model->vertexBuffer = device->createBuffer(&desc, agpuModel->vertices);
    }
    {
        model->vertexBinding = device->createVertexBinding(modelVertexLayout);
        model->vertexBinding->bindVertexBuffers(1, &model->vertexBuffer);
    }

    {
        agpu_buffer_description desc = {0};
        desc.size = agpu_size(indexBufferSize);
        desc.heap_type = AGPU_MEMORY_HEAP_TYPE_DEVICE_LOCAL;
        desc.usage_modes = agpu_buffer_usage_mask(AGPU_COPY_DESTINATION_BUFFER | AGPU_ELEMENT_ARRAY_BUFFER);
        desc.main_usage_mode = AGPU_ELEMENT_ARRAY_BUFFER;
        desc.mapping_flags = AGPU_MAP_DYNAMIC_STORAGE_BIT;
        desc.stride = 2;

        model->indexBuffer = device->createBuffer(&desc, agpuModel->indices);
    }

    return model;
}

agpu_texture_ref
Molevis::loadVRModelTexture(agpu_vr_render_model_texture *vrTexture)
{
    agpu_texture_description desc = {};
    desc.type = AGPU_TEXTURE_2D;
    desc.width = vrTexture->width;
    desc.height = vrTexture->height;
    desc.depth = 1;
    desc.layers = 1;
    desc.miplevels = 1;
    desc.format = AGPU_TEXTURE_FORMAT_B8G8R8A8_UNORM_SRGB;
    desc.usage_modes = agpu_texture_usage_mode_mask(AGPU_TEXTURE_USAGE_SAMPLED | AGPU_TEXTURE_USAGE_COPY_DESTINATION);
    desc.main_usage_mode = AGPU_TEXTURE_USAGE_SAMPLED;
    desc.heap_type = AGPU_MEMORY_HEAP_TYPE_DEVICE_LOCAL;
    desc.sample_count = 1;
    desc.sample_quality = 0;
    
    auto texture = device->createTexture(&desc);
    texture->uploadTextureData(0, 0, vrTexture->pitch, vrTexture->pitch*vrTexture->height, vrTexture->data);
    return texture;
}
#undef main

void
Molevis::updateAndRender(float delta)
{
    uiElementQuadBuffer.clear();

    // Left drag.
    if(hasLeftDragEvent && !hasHandledLeftDragEvent)
    {
        cameraAngle += Vector3(float(leftDragDeltaY), float(leftDragDeltaX), 0) * float(0.1f/M_PI);
        cameraAngle.x = std::min(std::max(cameraAngle.x, float(-M_PI*0.5)), float(M_PI*0.5));
    }
    cameraMatrix = Matrix3x3::XRotation(cameraAngle.x) * Matrix3x3::YRotation(cameraAngle.y);

    // Right drag.
    if(hasRightDragEvent && !hasHandledRightDragEvent)
    {
        cameraTranslation += cameraMatrix * (Vector3(float(rightDragDeltaX), float(-rightDragDeltaY), 0) * 0.1f);
    }

    // Mouse wheel.
    if(hasWheelEvent && !hasHandledWheelEvent)
    {
        cameraTranslation += cameraMatrix * Vector3(0, 0, -float(wheelDelta)*0.1f);
    }

    char buffer[64];
    snprintf(buffer, sizeof(buffer), "%d Atoms. %d Bonds. Sim iter %05d. Frame time %0.3f ms.", int(atomDescriptions.size()), int(atomBondDescriptions.size()), simulationIteration.load(), delta*1000.0);
    drawString(buffer, Vector2{5, 5}, Vector4{0.1f, 1.0f, 0.1f, 1.0f});

    auto cameraInverseMatrix = cameraMatrix.transposed();
    auto cameraInverseTranslation = cameraInverseMatrix * -cameraTranslation;
    float cameraFovY = 60.0;
    float cameraAspect = float(cameraState.screenWidth)/float(cameraState.screenHeight);

    cameraState.viewMatrix = Matrix4x4::withMatrix3x3AndTranslation(cameraInverseMatrix, cameraInverseTranslation);
    cameraState.inverseViewMatrix = cameraState.viewMatrix.inverse();
    cameraState.projectionMatrix = Matrix4x4::perspective(cameraFovY, cameraAspect, cameraState.nearDistance, cameraState.farDistance, device->hasTopLeftNdcOrigin());
    cameraState.inverseProjectionMatrix = cameraState.projectionMatrix.inverse();

    cameraState.atomModelMatrix = Matrix4x4::withMatrix3x3AndTranslation(Matrix3x3::withScale(modelScaleFactor), modelPosition);
    cameraState.atomInverseModelMatrix = cameraState.atomModelMatrix.inverse();

    cameraViewFrustum.makePerspective(60.0, cameraAspect, cameraState.nearDistance, cameraState.farDistance);
    cameraWorldFrustum = cameraViewFrustum.transformedWith(cameraState.inverseViewMatrix);
    cameraAtomFrustum = cameraWorldFrustum.transformedWith(cameraState.atomInverseModelMatrix);

    if(!hasLeftDragEvent && !hasRightDragEvent && !isVirtualReality && atomDescriptions.size() <= 10000)
    {
        auto normalizedPosition = Vector2(float(mousePositionX) / float(cameraState.screenWidth), 1.0f - float(mousePositionY) / float(cameraState.screenHeight));
        //printf("np %f %f\n", normalizedPosition.x, normalizedPosition.y);
        Ray frustumRay = cameraAtomFrustum.rayForNormalizedPoint(normalizedPosition);

        findHighlightedAtom(frustumRay);
    }

    hmdCameraState = cameraState;
    leftEyeCameraState = hmdCameraState;
    rightEyeCameraState = hmdCameraState;

    if(isVirtualReality)
        updateVRState();

    // Upload the data buffers.
    leftEyeCameraStateUniformBuffer->uploadBufferData(0, sizeof(leftEyeCameraState), &leftEyeCameraState);
    rightEyeCameraStateUniformBuffer->uploadBufferData(0, sizeof(rightEyeCameraState), &rightEyeCameraState);
    uiDataBuffer->uploadBufferData(0, agpu_size(uiElementQuadBuffer.size() * sizeof(UIElementQuad)), uiElementQuadBuffer.data());

    // Upload the new state
    {
        std::unique_lock l(renderingAtomStateMutex);
        if(renderingAtomStateDirty)
        {
            atomStateFrontBuffer->uploadBufferData(0, agpu_size(renderingAtomState.size()*sizeof(AtomState)), renderingAtomState.data());
            renderingAtomStateDirty = false;
        }
    }

    emitRenderCommands();
}


void
Molevis::findHighlightedAtom(const Ray &ray)
{
    int bestFound = -1;
    float bestFoundDistance = INFINITY;

    {
        std::unique_lock l(renderingAtomStateMutex);
        for(size_t i = 0; i < renderingAtomState.size(); ++i)
        {
            auto &atom = renderingAtomState[i];

            Sphere atomSphere;
            atomSphere.center = atom.position;
            atomSphere.radius = atomDescriptions[i].radius;

            Vector2 lambdas = {};
            if(atomSphere.rayIntersectionTest(ray, lambdas))
            {
                auto minLambda = std::min(lambdas.x, lambdas.y);
                if(minLambda < bestFoundDistance)
                {
                    bestFound = int(i);
                    bestFoundDistance = minLambda;
                }
            }
        }
    }

    currentHighlightedAtom = bestFound;
}

void
Molevis::updateVRState()
{
    vrSystem->waitAndFetchPoses();
    float nearDistance = cameraState.nearDistance;
    float farDistance = cameraState.farDistance;
    currentHighlightedAtom = -1;

    size_t poseCount = vrSystem->getCurrentTrackedDevicePoseCount();
    for(size_t i = 0; i < poseCount; ++i)
    {
        agpu_vr_tracked_device_pose trackedPose;
        //agpu_vr_tracked_device_pose renderTrackedPose;
        vrSystem->getCurrentTrackedDevicePoseInto(agpu_size(i), &trackedPose);
        if(!trackedPose.is_valid)
            continue;

        if(trackedPose.device_class == AGPU_VR_TRACKED_DEVICE_CLASS_CONTROLLER)
        {
            auto modelMatrix = Matrix4x4::fromAgpu(trackedPose.device_to_absolute_tracking);
            if(trackedPose.device_role == AGPU_VR_TRACKED_DEVICE_ROLE_LEFT_HAND)
            {
                auto & controller = handControllers[0];
                controller.modelState.modelMatrix = cameraState.inverseViewMatrix*modelMatrix;
                controller.modelState.inverseModelMatrix = modelMatrix.inverse();

                agpu_vr_controller_state controllerState;
                if(vrSystem->getControllerState(agpu_size(i), &controllerState))
                {
                    controller.convertState(controllerState);
                    if(controller.triggerAxisState.x > 0.5f)
                    {
                        Vector3 origin = (cameraState.atomInverseModelMatrix * (controller.modelState.modelMatrix * Vector4(0, 0, 0, 1))).xyz();
                        Vector3 direction = (cameraState.atomInverseModelMatrix * (controller.modelState.modelMatrix * Vector4(0, 0, -1, 0))).xyz().normalized();
                        auto ray = Ray::withOriginAndDirection(origin, direction);
                        //printf("Left origin %f %f %f direction %f %f %f\n",
                        //    origin.x, origin.y, origin.z,
                        //    direction.x, direction.y, direction.z);
                        findHighlightedAtom(ray);
                    }
                }

                if(!controller.modelStateBinding)
                {
                    agpu_buffer_description desc = {};
                    desc.size = (sizeof(ModelState) + 255) & (-256);
                    desc.heap_type = AGPU_MEMORY_HEAP_TYPE_HOST_TO_DEVICE;
                    desc.usage_modes = agpu_buffer_usage_mask(AGPU_COPY_DESTINATION_BUFFER | AGPU_UNIFORM_BUFFER);
                    desc.main_usage_mode = AGPU_UNIFORM_BUFFER;
                    desc.mapping_flags = AGPU_MAP_DYNAMIC_STORAGE_BIT;
                    controller.modelStateBuffer = device->createBuffer(&desc, nullptr);

                    controller.modelStateBinding = shaderSignature->createShaderResourceBinding(4);
                    controller.modelStateBinding->bindUniformBuffer(0, controller.modelStateBuffer);
                }
                controller.modelStateBuffer->uploadBufferData(0, sizeof(controller.modelState), &controller.modelState);

                if(!controller.deviceModel)
                {
                    auto model = vrSystem->getTrackedDeviceRenderModel(agpu_size(i));
                    if(model && model->texture)
                    {
                        controller.deviceModel = loadDeviceModel(model);
                        auto texture = loadVRModelTexture(model->texture);
                        controller.modelStateBinding->bindSampledTextureView(1, texture->getOrCreateFullView());
                    }
                }

            }
            else if(trackedPose.device_role == AGPU_VR_TRACKED_DEVICE_ROLE_RIGHT_HAND)
            {
                auto & controller = handControllers[1];
                controller.modelState.modelMatrix = cameraState.inverseViewMatrix*modelMatrix;
                controller.modelState.inverseModelMatrix = modelMatrix.inverse();

                agpu_vr_controller_state controllerState;
                if(vrSystem->getControllerState(agpu_size(i), &controllerState))
                {
                    controller.convertState(controllerState);
                    if(controller.triggerAxisState.x > 0.5f)
                    {
                        Vector3 origin = (cameraState.atomInverseModelMatrix * (controller.modelState.modelMatrix * Vector4(0, 0, 0, 1))).xyz();
                        Vector3 direction = (cameraState.atomInverseModelMatrix * (controller.modelState.modelMatrix * Vector4(0, 0, -1, 0))).xyz().normalized();
                        auto ray = Ray::withOriginAndDirection(origin, direction);
                        //printf("Right origin %f %f %f direction %f %f %f\n",
                        //    origin.x, origin.y, origin.z,
                        //    direction.x, direction.y, direction.z);
                        findHighlightedAtom(ray);
                    }
                }

                if(!controller.modelStateBinding)
                {
                    agpu_buffer_description desc = {};
                    desc.size = (sizeof(ModelState) + 255) & (-256);
                    desc.heap_type = AGPU_MEMORY_HEAP_TYPE_HOST_TO_DEVICE;
                    desc.usage_modes = agpu_buffer_usage_mask(AGPU_COPY_DESTINATION_BUFFER | AGPU_UNIFORM_BUFFER);
                    desc.main_usage_mode = AGPU_UNIFORM_BUFFER;
                    desc.mapping_flags = AGPU_MAP_DYNAMIC_STORAGE_BIT;
                    controller.modelStateBuffer = device->createBuffer(&desc, nullptr);

                    controller.modelStateBinding = shaderSignature->createShaderResourceBinding(4);
                    controller.modelStateBinding->bindUniformBuffer(0, controller.modelStateBuffer);
                }
                controller.modelStateBuffer->uploadBufferData(0, sizeof(controller.modelState), &controller.modelState);

                if(!controller.deviceModel)
                {
                    auto model = vrSystem->getTrackedDeviceRenderModel(agpu_size(i));
                    if(model && model->texture)
                    {
                        controller.deviceModel = loadDeviceModel(model);
                        auto texture = loadVRModelTexture(model->texture);
                        controller.modelStateBinding->bindSampledTextureView(1, texture->getOrCreateFullView());
                    }
                }
            }
        }
        else if(trackedPose.device_class == AGPU_VR_TRACKED_DEVICE_CLASS_HMD)
        {
            auto headMatrix = Matrix4x4::fromAgpu(trackedPose.device_to_absolute_tracking);
            hmdCameraState.viewMatrix = headMatrix.inverse()*cameraState.viewMatrix;
            hmdCameraState.inverseViewMatrix = hmdCameraState.viewMatrix.inverse();
        }
    }

    leftEyeCameraState = hmdCameraState;
    rightEyeCameraState = hmdCameraState;

    // Left eye
    {
        agpu_frustum_tangents tangents;
        vrSystem->getProjectionFrustumTangents(AGPU_VR_EYE_LEFT, &tangents);
        leftEyeCameraState.projectionMatrix = Matrix4x4::frustum(tangents.left*nearDistance, tangents.right*nearDistance, tangents.bottom*nearDistance, tangents.top*nearDistance, nearDistance, farDistance, device->hasTopLeftNdcOrigin());
        leftEyeCameraState.inverseProjectionMatrix = leftEyeCameraState.projectionMatrix.inverse();

        agpu_matrix4x4f eyeToHead;
        vrSystem->getEyeToHeadTransform(AGPU_VR_EYE_LEFT, &eyeToHead);

        auto eyeToHeadMatrix = Matrix4x4::fromAgpu(eyeToHead);
        leftEyeCameraState.viewMatrix = eyeToHeadMatrix.inverse() * leftEyeCameraState.viewMatrix;
        leftEyeCameraState.inverseViewMatrix = leftEyeCameraState.viewMatrix.inverse();
    }

    // right eye
    {
        agpu_frustum_tangents tangents;
        vrSystem->getProjectionFrustumTangents(AGPU_VR_EYE_RIGHT, &tangents);
        rightEyeCameraState.projectionMatrix = Matrix4x4::frustum(tangents.left*nearDistance, tangents.right*nearDistance, tangents.bottom*nearDistance, tangents.top*nearDistance, nearDistance, farDistance, device->hasTopLeftNdcOrigin());
        rightEyeCameraState.inverseProjectionMatrix = rightEyeCameraState.projectionMatrix.inverse();

        agpu_matrix4x4f eyeToHead;
        vrSystem->getEyeToHeadTransform(AGPU_VR_EYE_RIGHT, &eyeToHead);

        auto eyeToHeadMatrix = Matrix4x4::fromAgpu(eyeToHead);
        rightEyeCameraState.viewMatrix = eyeToHeadMatrix.inverse() * rightEyeCameraState.viewMatrix;
        rightEyeCameraState.inverseViewMatrix = rightEyeCameraState.viewMatrix.inverse();
    }
}

void
Molevis::emitRenderCommands()
{    
    // Build the command list
    commandAllocator->reset();
    commandList->reset(commandAllocator, nullptr);

    commandList->setShaderSignature(shaderSignature);
    commandList->useComputeShaderResources(samplersBinding);
    commandList->useComputeShaderResources(leftEyeScreenStateBinding);
    commandList->useComputeShaderResources(atomFrontBufferBinding);
    commandList->useComputeShaderResources(atomBoundQuadBufferBinding);
    
    PushConstants pushConstants = {};
    pushConstants.highlighedAtom = currentHighlightedAtom;
    commandList->pushConstants(0, sizeof(pushConstants), &pushConstants);

    if(isStereo || isVirtualReality)
    {
        emitCommandsForEyeRendering(false);
        emitCommandsForEyeRendering(true);
    }
    else
    {
        emitCommandsForEyeRendering(false);
    }

    auto outputBackBuffer = swapChain->getCurrentBackBuffer();
    commandList->beginRenderPass(outputRenderPass, outputBackBuffer, false);
    commandList->setViewport(0, 0, displayWidth, displayHeight);
    commandList->setScissor(0, 0, displayWidth, displayHeight);

    if(isStereo)
    {
        // Side by side compose
        commandList->usePipelineState(sideBySidePipeline);
        commandList->drawArrays(3, 1, 0, 0);
    }
    else
    {
        // Pass the left eye output
        commandList->usePipelineState(passthroughPipeline);
        commandList->drawArrays(3, 1, 0, 0);
    }
    
    commandList->endRenderPass();

    commandList->close();

    // Queue the command list
    commandQueue->addCommandList(commandList);

    swapBuffers();
    commandQueue->finishExecution();

    if(isVirtualReality && vrSystem)
        vrSystem->submitEyeRenderTargets(leftEyeTexture, rightEyeTexture);        
}

void Molevis::emitCommandsForEyeRendering(bool isRightEye)
{
    if(isRightEye)
        commandList->useComputeShaderResources(rightEyeScreenStateBinding);
    else
        commandList->useComputeShaderResources(leftEyeScreenStateBinding);

    // Screen bounding quad computations.
    commandList->usePipelineState(atomScreenQuadBufferComputationPipeline);
    commandList->dispatchCompute(agpu_uint((atomDescriptions.size() + 31)/32), 1, 1);
    commandList->memoryBarrier(AGPU_PIPELINE_STAGE_COMPUTE_SHADER, AGPU_PIPELINE_STAGE_VERTEX_SHADER, AGPU_ACCESS_SHADER_WRITE, AGPU_ACCESS_SHADER_READ);

    commandList->beginRenderPass(mainRenderPass, hdrTargetFramebuffer, false);
    commandList->setViewport(0, 0, framebufferDisplayWidth, framebufferDisplayHeight);
    commandList->setScissor(0, 0, framebufferDisplayWidth, framebufferDisplayHeight);

    // Atoms
    commandList->usePipelineState(atomDrawPipeline);
    commandList->useShaderResources(samplersBinding);
    if(isRightEye)
        commandList->useShaderResources(rightEyeScreenStateBinding);
    else
        commandList->useShaderResources(leftEyeScreenStateBinding);
    commandList->useShaderResources(atomFrontBufferBinding);
    commandList->useShaderResources(atomBoundQuadBufferBinding);
    commandList->drawArrays(4, agpu_uint(atomDescriptions.size()), 0, 0);

    // Bonds
    if(bondXRay)
        commandList->usePipelineState(bondXRayDrawPipeline);
    else
        commandList->usePipelineState(bondDrawPipeline);
    commandList->drawArrays(4, agpu_uint(atomBondDescriptions.size()), 0, 0);

    // Floor grid
    commandList->usePipelineState(floorGridDrawPipeline);
    commandList->drawArrays(4, 1, 0, 0);

    // Hand controllers
    for(auto &handController : handControllers)
    {
        if(!handController.deviceModel)
            continue;

        commandList->useShaderResources(handController.modelStateBinding);
        commandList->usePipelineState(modelPipelineState);
        commandList->useVertexBinding(handController.deviceModel->vertexBinding);
        commandList->useIndexBuffer(handController.deviceModel->indexBuffer);
        commandList->drawElements(agpu_uint(handController.deviceModel->indexCount), 1, 0, 0, 0);


        if(pointerModel && handController.triggerAxisState.x >= 0.5f)
        {
            commandList->usePipelineState(pointerModelPipelineState);
            commandList->useVertexBinding(pointerModel->vertexBinding);
            commandList->useIndexBuffer(pointerModel->indexBuffer);
            commandList->drawElements(agpu_uint(pointerModel->indexCount), 1, 0, 0, 0);
        }
    }

    // Finish the hdr rendering
    commandList->endRenderPass();

    // Tone mapping output generation
    if(isRightEye)
        commandList->beginRenderPass(outputRenderPass, rightEyeFramebuffer, false);
    else
        commandList->beginRenderPass(outputRenderPass, leftEyeFramebuffer, false);
    commandList->setViewport(0, 0, framebufferDisplayWidth, framebufferDisplayHeight);
    commandList->setScissor(0, 0, framebufferDisplayWidth, framebufferDisplayHeight);

    commandList->usePipelineState(filmicTonemappingPipeline);
    commandList->drawArrays(3, 1, 0, 0);

    // UI element pipeline
    commandList->usePipelineState(uiPipeline);
    commandList->drawArrays(4, agpu_uint(uiElementQuadBuffer.size()), 0, 0);
    
    commandList->endRenderPass();
}

void
Molevis::swapBuffers()
{
    auto errorCode = agpuSwapBuffers(swapChain.get());
    if(!errorCode)
        return;

    if(errorCode == AGPU_OUT_OF_DATE)
        recreateSwapChain();
}

agpu_texture_ref
Molevis::loadTexture(const char *fileName, bool nonColorData)
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

int main(int argc, const char **argv)
{
    return Molevis().mainStart(argc, argv);
}
