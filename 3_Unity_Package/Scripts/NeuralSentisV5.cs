using UnityEngine;
// ─────────────────────────────────────────────────────────────────────────────
// COMPATIBILIDAD DE VERSIONES DE SENTIS:
//   • Sentis 2.x / Unity Inference Engine (Unity 6+) → usa Unity.InferenceEngine
//   • Sentis 1.x (Unity 2022-2023 LTS)              → usa Unity.Sentis
//
// Si ves errores CS0234/CS0246, ve a:
//   Window → Package Manager → busca "Sentis" o "Inference Engine"
//   Si tienes Sentis 1.x: añade el símbolo SENTIS_1X en:
//   Project Settings → Player → Other Settings → Scripting Define Symbols
// ─────────────────────────────────────────────────────────────────────────────
#if SENTIS_1X
using Unity.Sentis;
#else
using Unity.InferenceEngine;
#endif
using UnityEngine.InputSystem;
using System.Collections;
using UnityEngine.Rendering;
using UnityEngine.Experimental.Rendering;


public class NeuralSentisV5 : MonoBehaviour
{
    public ModelAsset onnxModel;
    public ComputeShader pack7ChannelCompute;
    public Material temporalWarpMaterial;
    
    [Header("Inputs")]
    public RTHandle currentRGB;
    public RTHandle currentDepth;
    public RTHandle motionVectors;
    
    [Header("Internal Memory")]
    private RTHandle prevStyledOutput;
    private RTHandle warpedPrevOutput;
    
    private Worker worker;
    private Tensor<float> inputTensor;
    private ComputeTensorData computeTensorData;

    public enum ResolutionLevel { LowRes_480p, MidRes_720p, Native_512x512 }
    public ResolutionLevel currentResolution = ResolutionLevel.LowRes_480p;

    private int width = 480;
    private int height = 270;
    private bool isInitialized = false;
    private bool isInferenceRunning = false;
    
    public bool effectActive = true;
    public enum DebugMode { AI, Depth, RGB }
    public bool firstInferenceDone = false;
    public DebugMode debugMode = DebugMode.AI;

    void Start()
    {
        Debug.Log("<color=cyan>[Neural-Ink] Restaurando estabilidad y optimizando...</color>");
        var model = ModelLoader.Load(onnxModel);
        worker = new Worker(model, BackendType.GPUCompute);

        InitializeMemory();
        StartCoroutine(InferenceRoutine());
    }

    public void InitializeMemory()
    {
        // Si ya está inicializado, liberamos memoria antes de recrear (para cambio de resolución)
        if (isInitialized) 
        {
            worker?.Dispose();
            inputTensor?.Dispose();
            computeTensorData?.Dispose();
            currentRGB?.Release();
            currentDepth?.Release();
            motionVectors?.Release();
            prevStyledOutput?.Release();
            warpedPrevOutput?.Release();
        }

        switch (currentResolution)
        {
            case ResolutionLevel.LowRes_480p: width = 480; height = 270; break;
            case ResolutionLevel.MidRes_720p: width = 640; height = 360; break;
            case ResolutionLevel.Native_512x512: width = 512; height = 512; break;
        }

        currentRGB = RTHandles.Alloc(width, height, colorFormat: GraphicsFormat.R8G8B8A8_UNorm, name: "NeuralRGB");
        currentDepth = RTHandles.Alloc(width, height, colorFormat: GraphicsFormat.R16_UNorm, depthBufferBits: DepthBits.None, name: "NeuralDepth");
        motionVectors = RTHandles.Alloc(width, height, colorFormat: GraphicsFormat.R16G16_SFloat, name: "NeuralMotion");
        
        prevStyledOutput = RTHandles.Alloc(width, height, colorFormat: GraphicsFormat.R8G8B8A8_UNorm, name: "NeuralPrevOutput");
        warpedPrevOutput = RTHandles.Alloc(width, height, colorFormat: GraphicsFormat.R8G8B8A8_UNorm, name: "NeuralWarped");
        
        currentRGB.rt.Create();
        currentDepth.rt.Create();
        motionVectors.rt.Create();
        prevStyledOutput.rt.Create();
        warpedPrevOutput.rt.Create();

        Graphics.Blit(Texture2D.blackTexture, prevStyledOutput);

        var model = ModelLoader.Load(onnxModel);
        worker = new Worker(model, BackendType.GPUCompute);

        TensorShape shape = new TensorShape(1, 7, height, width);
        computeTensorData = new ComputeTensorData(shape.length);
        inputTensor = new Tensor<float>(shape, computeTensorData);

        isInitialized = true;
    }

    public void ChangeResolution(ResolutionLevel newLevel)
    {
        if (currentResolution == newLevel) return;
        currentResolution = newLevel;
        isInitialized = false; // Forzar reinicialización en el próximo frame
        InitializeMemory();
        Debug.Log($"<color=orange>[Neural-Ink]</color> Resolución cambiada a: {width}x{height}");
    }

    IEnumerator InferenceRoutine()
    {
        while (true)
        {
            if (isInitialized && effectActive && !isInferenceRunning)
            {
                yield return ExecuteInferenceStep();
                firstInferenceDone = true;
            }
            yield return null;
        }
    }

    IEnumerator ExecuteInferenceStep()
    {
        isInferenceRunning = true;

        // 1. Warping Temporal con Neighbor Clamping (Anti-Olas)
        temporalWarpMaterial.SetTexture("_MotionTex", motionVectors);
        temporalWarpMaterial.SetTexture("_CurrentTex", currentRGB); // FIX: Necesario para neighbor clamping
        Graphics.Blit(prevStyledOutput, warpedPrevOutput, temporalWarpMaterial);

        // 2. Packing (Sincronizado)
        int kernel = pack7ChannelCompute.FindKernel("PackInput");
        pack7ChannelCompute.SetTexture(kernel, "RGBTexture", currentRGB);
        pack7ChannelCompute.SetTexture(kernel, "DepthTexture", currentDepth);
        pack7ChannelCompute.SetTexture(kernel, "PrevWarpedTexture", warpedPrevOutput);
        pack7ChannelCompute.SetBuffer(kernel, "TensorBuffer", computeTensorData.buffer);
        pack7ChannelCompute.SetInt("width", width);
        pack7ChannelCompute.SetInt("height", height);

        int threadGroupsX = Mathf.CeilToInt(width / 8.0f);
        int threadGroupsY = Mathf.CeilToInt(height / 8.0f);
        pack7ChannelCompute.Dispatch(kernel, threadGroupsX, threadGroupsY, 1);

        float startTime = Time.realtimeSinceStartup;

        // 3. Inferencia
        worker.Schedule(inputTensor);

        // 4. ESPERA OPTIMIZADA: Dejamos que la GPU trabaje asíncronamente
        yield return null;

        // 5. Recoger resultado directamente al buffer original
        var outputTensor = worker.PeekOutput() as Tensor<float>;
        if (outputTensor != null)
        {
            TextureConverter.RenderToTexture(outputTensor, prevStyledOutput);
        }

        // Benchmark de Latencia Real (Pipeline Completo)
        float ms = (Time.realtimeSinceStartup - startTime) * 1000f;
        if (Time.frameCount % 60 == 0) // Un log por segundo para no ahogar la consola
        {
            Debug.Log($"<color=yellow>[Neural-Ink Benchmark]</color> Latencia Sentis (GPU Async): {ms:0.00} ms");
        }

        isInferenceRunning = false;
    }

    void Update()
    {
        if (Keyboard.current == null) return;
        if (Keyboard.current.nKey.wasPressedThisFrame) effectActive = !effectActive;
        if (Keyboard.current.mKey.wasPressedThisFrame) debugMode = (DebugMode)(((int)debugMode + 1) % 3);
        
        // Tecla 'L' para rotar resoluciones (Pervasive AI Scalability Demo)
        if (Keyboard.current.lKey.wasPressedThisFrame)
        {
            ResolutionLevel nextRes = (ResolutionLevel)(((int)currentResolution + 1) % 3);
            ChangeResolution(nextRes);
        }
    }

    void OnGUI()
    {
        if (!effectActive) return;

        GUIStyle style = new GUIStyle();
        style.fontSize = 18;
        style.normal.textColor = Color.white;
        Rect rect = new Rect(20, 20, 400, 30);
        
        string resName = currentResolution.ToString().Replace("_", " ");
        string text = $"<b><color=cyan>[Neural-Ink V5]</color></b> Mode: {debugMode} | Res: {width}x{height} ({resName})";
        
        GUI.Box(new Rect(10, 10, 420, 50), "");
        GUI.Label(rect, text, style);
        GUI.Label(new Rect(20, 40, 400, 20), "Press 'L' to Scale AI | 'M' for Debug | 'N' to Toggle", new GUIStyle { fontSize = 12, normal = new UnityEngine.GUIStyleState { textColor = Color.gray } });
    }

    public RTHandle GetOutputRTHandle()
    {
        if (!isInitialized) return null;
        switch (debugMode)
        {
            case DebugMode.AI: return prevStyledOutput;
            case DebugMode.Depth: return currentDepth;
            case DebugMode.RGB: return currentRGB;
            default: return prevStyledOutput;
        }
    }

    void OnDestroy()
    {
        worker?.Dispose();
        inputTensor?.Dispose();
        computeTensorData?.Dispose();
    }
}
