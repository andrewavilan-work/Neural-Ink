using UnityEngine;
using UnityEngine.Rendering;
using UnityEngine.Rendering.Universal;
using UnityEngine.InputSystem;
using System.IO;
using UnityEngine.Experimental.Rendering;

[ExecuteInEditMode]
public class NeuralCapture : MonoBehaviour
{
    [Header("Configuración de Captura")]
    public string folderName = "NeuralDataset_V5";
    // 24 FPS: sweet spot para temporal consistency.
    // Los motion vectors de Unity son por render-frame — capturar a < 15 FPS
    // hace que el optical flow represente solo una fracción del desplazamiento real
    // entre frames capturados → el warp en training queda desalineado → parpadeo.
    // A 24 FPS y juego a 60 FPS, el gap es ~2-3 render-frames: coherente y preciso.
    public float capturesPerSecond = 24f;

    // Resolución fija de captura — debe coincidir con IMG_SIZE del training MI300X
    private const int CAPTURE_WIDTH  = 768;
    private const int CAPTURE_HEIGHT = 768;

    private bool isCapturing = false;
    private int frameIndex = 0;
    private string pathRGB, pathDepth, pathMotion;
    private string pathBG_RGB, pathBG_Depth;
    private float nextCaptureTime = 0f;
    private float sessionStartTime = 0f;  // Para calcular tasa real de captura
    
    // Usamos RTHandles para que sean automáticos y dinámicos
    [HideInInspector] public RTHandle internalRGB;
    [HideInInspector] public RTHandle internalDepth;
    [HideInInspector] public RTHandle internalMotion;
    [HideInInspector] public RTHandle backgroundRGB;
    [HideInInspector] public RTHandle backgroundDepth;

    private int lastWidth, lastHeight;

    void OnEnable()
    {
        SetupFolders();
        Debug.Log($"<color=cyan>[Neural-Ink] Captura Automática Activada.</color>");
    }

    void SetupFolders()
    {
        string baseDir = Path.Combine(Directory.GetParent(Application.dataPath).FullName, folderName);
        pathRGB = Path.Combine(baseDir, "rgb");
        pathDepth = Path.Combine(baseDir, "depth");
        pathMotion = Path.Combine(baseDir, "motion");
        pathBG_RGB = Path.Combine(baseDir, "bg_rgb");
        pathBG_Depth = Path.Combine(baseDir, "bg_depth");

        if (!Directory.Exists(pathRGB)) Directory.CreateDirectory(pathRGB);
        if (!Directory.Exists(pathDepth)) Directory.CreateDirectory(pathDepth);
        if (!Directory.Exists(pathMotion)) Directory.CreateDirectory(pathMotion);
        if (!Directory.Exists(pathBG_RGB)) Directory.CreateDirectory(pathBG_RGB);
        if (!Directory.Exists(pathBG_Depth)) Directory.CreateDirectory(pathBG_Depth);
    }

    public bool IsCapturing() => isCapturing && Application.isPlaying && Time.time >= nextCaptureTime;

    // Siempre usamos CAPTURE_WIDTH x CAPTURE_HEIGHT (768x768)
    // Ignoramos la resolución de pantalla para que el dataset sea consistente
    // con el training MI300X (IMG_SIZE=768)
    public void CheckResolution(int w, int h)
    {
        if (lastWidth != CAPTURE_WIDTH || lastHeight != CAPTURE_HEIGHT)
        {
            lastWidth  = CAPTURE_WIDTH;
            lastHeight = CAPTURE_HEIGHT;
            ReallocateBuffers(CAPTURE_WIDTH, CAPTURE_HEIGHT);
        }
    }

    void ReallocateBuffers(int w, int h)
    {
        RTHandles.Release(internalRGB);
        RTHandles.Release(internalDepth);
        RTHandles.Release(internalMotion);
        RTHandles.Release(backgroundRGB);
        RTHandles.Release(backgroundDepth);

        // Forzar siempre 768x768 — ignorar w/h de pantalla
        internalRGB     = RTHandles.Alloc(CAPTURE_WIDTH, CAPTURE_HEIGHT, colorFormat: GraphicsFormat.R8G8B8A8_SRGB,  name: "CapFullRGB");
        internalDepth   = RTHandles.Alloc(CAPTURE_WIDTH, CAPTURE_HEIGHT, colorFormat: GraphicsFormat.R16_UNorm,      name: "CapFullDepth");
        internalMotion  = RTHandles.Alloc(CAPTURE_WIDTH, CAPTURE_HEIGHT, colorFormat: GraphicsFormat.R16G16_SFloat,  name: "CapFullMotion");
        backgroundRGB   = RTHandles.Alloc(CAPTURE_WIDTH, CAPTURE_HEIGHT, colorFormat: GraphicsFormat.R8G8B8A8_SRGB,  name: "CapBGRGB");
        backgroundDepth = RTHandles.Alloc(CAPTURE_WIDTH, CAPTURE_HEIGHT, colorFormat: GraphicsFormat.R16_UNorm,      name: "CapBGDepth");

        Debug.Log($"<color=cyan>[Neural-Ink] Buffers allocados: {CAPTURE_WIDTH}x{CAPTURE_HEIGHT} (fijo para dataset MI300X)</color>");
    }

    void Update()
    {
        if (Keyboard.current != null && Keyboard.current.xKey.wasPressedThisFrame)
        {
            isCapturing = !isCapturing;
            if (isCapturing)
            {
                sessionStartTime = Time.time;
                Debug.Log($"<color=green>[Neural-Ink] CAPTURA INICIADA | Res: {CAPTURE_WIDTH}x{CAPTURE_HEIGHT} | Target: {capturesPerSecond} fps | Total acumulado: {frameIndex}</color>");
            }
            else
            {
                float elapsed  = Time.time - sessionStartTime;
                float realFps  = elapsed > 0 ? frameIndex / elapsed : 0;
                Debug.Log($"<color=red>[Neural-Ink] CAPTURA DETENIDA | Total frames: {frameIndex} | Duracion: {elapsed:F1}s | FPS real: {realFps:F2}</color>");
            }
        }
    }

    public void NotifyFrameCaptured()
    {
        nextCaptureTime = Time.time + (1f / capturesPerSecond);
        int currentFrame = frameIndex;

        // Captura asíncrona a 768x768 (CAPTURE_WIDTH x CAPTURE_HEIGHT)
        RequestCapture(internalRGB,     Path.Combine(pathRGB,     $"frame_{currentFrame:D5}.png"), TextureFormat.RGBA32);
        RequestCapture(internalDepth,   Path.Combine(pathDepth,   $"frame_{currentFrame:D5}.png"), TextureFormat.R16);
        RequestCapture(internalMotion,  Path.Combine(pathMotion,  $"frame_{currentFrame:D5}.exr"), TextureFormat.RGHalf, true);
        RequestCapture(backgroundRGB,   Path.Combine(pathBG_RGB,  $"frame_{currentFrame:D5}.png"), TextureFormat.RGBA32);
        RequestCapture(backgroundDepth, Path.Combine(pathBG_Depth,$"frame_{currentFrame:D5}.png"), TextureFormat.R16);

        frameIndex++;

        // Log en consola cada 10 capturas — Target: 2500 frames (~5.2 GB para AMD Cloud)
        if (frameIndex % 10 == 0)
        {
            const int TARGET_FRAMES = 2500;
            float elapsed   = Time.time - sessionStartTime;
            float realFps   = elapsed > 0f ? frameIndex / elapsed : 0f;
            float remaining = realFps > 0f ? (TARGET_FRAMES - frameIndex) / realFps : 0f;
            float pct       = (frameIndex / (float)TARGET_FRAMES) * 100f;
            string bar      = new string('#', Mathf.FloorToInt(pct / 5f))
                            + new string('-', 20 - Mathf.FloorToInt(pct / 5f));

            Debug.Log($"<color=lime>[Neural-Ink Dataset]</color> "
                    + $"[{bar}] {pct:F0}% | "
                    + $"<b>{frameIndex}/{TARGET_FRAMES}</b> | "
                    + $"{CAPTURE_WIDTH}x{CAPTURE_HEIGHT} | "
                    + $"FPS: {realFps:F1} | "
                    + $"Restante: {(frameIndex < TARGET_FRAMES ? $"{remaining:F0}s" : "COMPLETO")}");

            if (frameIndex % 500 == 0 && frameIndex > 0)
                Debug.Log($"<color=yellow>[Neural-Ink] HITO {frameIndex} frames | ~{frameIndex * 2.1f:F0} MB | {TARGET_FRAMES - frameIndex} para completar</color>");

            if (frameIndex >= TARGET_FRAMES)
                Debug.Log($"<color=cyan>[Neural-Ink] DATASET COMPLETO: {frameIndex} frames | ~{frameIndex * 2.1f / 1024f:F1} GB | Presiona X para detener y subir al AMD Cloud.</color>");
        }
    }

    void RequestCapture(RTHandle rt, string fullPath, TextureFormat format, bool isExr = false)
    {
        if (rt == null || rt.rt == null) return;
        AsyncGPUReadback.Request(rt.rt, 0, format, (request) => {
            if (!request.hasError) SaveImage(request.GetData<byte>().ToArray(), fullPath, format, isExr, rt.rt.width, rt.rt.height);
        });
    }

    void SaveImage(byte[] data, string fullPath, TextureFormat format, bool isExr, int w, int h)
    {
        Texture2D tex = new Texture2D(w, h, format, false);
        if (data.Length >= (w * h * GetBytesPerPixel(format)))
        {
            tex.LoadRawTextureData(data);
            tex.Apply();
            if (isExr) File.WriteAllBytes(fullPath, tex.EncodeToEXR(Texture2D.EXRFlags.None));
            else File.WriteAllBytes(fullPath, tex.EncodeToPNG());
        }
        DestroyImmediate(tex);
    }

    int GetBytesPerPixel(TextureFormat format)
    {
        switch (format)
        {
            case TextureFormat.R16: return 2;
            case TextureFormat.RGHalf: return 4;
            case TextureFormat.RGBA32: return 4;
            default: return 4;
        }
    }

    void OnDisable()
    {
        RTHandles.Release(internalRGB);
        RTHandles.Release(internalDepth);
        RTHandles.Release(internalMotion);
        RTHandles.Release(backgroundRGB);
        RTHandles.Release(backgroundDepth);
    }
}
