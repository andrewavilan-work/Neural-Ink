using UnityEngine;
using UnityEngine.Rendering;
using UnityEngine.Rendering.Universal;
using UnityEngine.Rendering.RenderGraphModule;

public class NeuralCaptureFeature : ScriptableRendererFeature
{
    class NeuralCapturePass : ScriptableRenderPass
    {
        public NeuralCapture controller;
        public Material depthConvertMaterial; // Material para Linear01Depth

        public override void RecordRenderGraph(RenderGraph renderGraph, ContextContainer frameData)
        {
            if (controller == null || !controller.IsCapturing()) return;

            UniversalResourceData resourceData = frameData.Get<UniversalResourceData>();
            UniversalCameraData cameraData = frameData.Get<UniversalCameraData>();
            var additionalData = cameraData.camera.GetUniversalAdditionalCameraData();

            controller.CheckResolution(cameraData.cameraTargetDescriptor.width, cameraData.cameraTargetDescriptor.height);

            TextureHandle cameraColor = resourceData.activeColorTexture;
            TextureHandle cameraDepth = resourceData.cameraDepthTexture;
            
            // Fallback: Si el buffer oficial no es válido, intentamos el activo
            if (!cameraDepth.IsValid()) cameraDepth = resourceData.activeDepthTexture;

            TextureHandle cameraMotion = resourceData.motionVectorColor;

            if (!cameraColor.IsValid()) return;

            bool isOverlay = additionalData.renderType == CameraRenderType.Overlay;
            bool isBase = additionalData.renderType == CameraRenderType.Base;
            
            bool hasStack = false;
            if (isBase && additionalData.cameraStack != null)
            {
                hasStack = additionalData.cameraStack.Count > 0;
            }

            // CASO A: Captura de Fondo (Background)
            if (isBase)
            {
                if (controller.backgroundRGB != null)
                {
                    TextureHandle destBG_RGB = renderGraph.ImportTexture(controller.backgroundRGB);
                    AddBlitPass(renderGraph, cameraColor, destBG_RGB, "Capture BG Color", null);
                }
                
                if (cameraDepth.IsValid() && controller.backgroundDepth != null)
                {
                    TextureHandle destBG_Depth = renderGraph.ImportTexture(controller.backgroundDepth);
                    AddBlitPass(renderGraph, cameraDepth, destBG_Depth, "Capture BG Depth", depthConvertMaterial);
                }

                if (!hasStack) 
                {
                    // Compatibilidad 1 Cámara: Copiamos también a los buffers finales
                    if (controller.internalRGB != null)
                        AddBlitPass(renderGraph, cameraColor, renderGraph.ImportTexture(controller.internalRGB), "Capture Final Color", null);
                    
                    if (cameraDepth.IsValid() && controller.internalDepth != null)
                        AddBlitPass(renderGraph, cameraDepth, renderGraph.ImportTexture(controller.internalDepth), "Capture Final Depth", depthConvertMaterial);
                        
                    if (cameraMotion.IsValid() && controller.internalMotion != null)
                        AddBlitPass(renderGraph, cameraMotion, renderGraph.ImportTexture(controller.internalMotion), "Capture Motion", null);

                    controller.NotifyFrameCaptured();
                }
            }

            // CASO B: Captura Final (Full)
            if (isOverlay)
            {
                if (controller.internalRGB != null)
                {
                    TextureHandle destRGB = renderGraph.ImportTexture(controller.internalRGB);
                    AddBlitPass(renderGraph, cameraColor, destRGB, "Capture Final Color", null);
                }

                if (cameraDepth.IsValid() && controller.internalDepth != null)
                {
                    TextureHandle destDepth = renderGraph.ImportTexture(controller.internalDepth);
                    AddBlitPass(renderGraph, cameraDepth, destDepth, "Capture Final Depth", depthConvertMaterial);
                }
                
                if (cameraMotion.IsValid() && controller.internalMotion != null)
                {
                    TextureHandle destMotion = renderGraph.ImportTexture(controller.internalMotion);
                    AddBlitPass(renderGraph, cameraMotion, destMotion, "Capture Motion", null);
                }

                controller.NotifyFrameCaptured();
            }
        }

        void AddBlitPass(RenderGraph graph, TextureHandle src, TextureHandle dst, string name, Material mat)
        {
            using (var builder = graph.AddRasterRenderPass<PassData>(name, out var passData))
            {
                passData.src = src;
                passData.mat = mat;
                builder.UseTexture(src, AccessFlags.Read);
                builder.SetRenderAttachment(dst, 0, AccessFlags.Write);
                builder.AllowPassCulling(false);
                builder.SetRenderFunc((PassData data, RasterGraphContext context) =>
                {
                    if (data.mat != null)
                        Blitter.BlitTexture(context.cmd, data.src, new Vector4(1, 1, 0, 0), data.mat, 0);
                    else
                        Blitter.BlitTexture(context.cmd, data.src, new Vector4(1, 1, 0, 0), 0f, false);
                });
            }
        }

        private class PassData { public TextureHandle src; public Material mat; }
    }

    public Material depthConvertMaterial;
    NeuralCapturePass m_ScriptablePass;

    public override void Create()
    {
        m_ScriptablePass = new NeuralCapturePass();
        m_ScriptablePass.renderPassEvent = RenderPassEvent.BeforeRenderingPostProcessing;

        // Auto-configuración comercial: si no hay material, buscamos el shader
        if (depthConvertMaterial == null)
        {
            Shader s = Shader.Find("Hidden/Neural/LinearDepthCapture");
            if (s != null) 
            {
                depthConvertMaterial = CoreUtils.CreateEngineMaterial(s);
                Debug.Log("<color=green>[Neural-Ink] Shader de Profundidad encontrado y configurado.</color>");
            }
            else
            {
                Debug.LogError("<color=red>[Neural-Ink] ¡ERROR! No se encuentra el shader 'Hidden/Neural/LinearDepthCapture'.</color>");
            }
        }
    }

    public override void AddRenderPasses(ScriptableRenderer renderer, ref RenderingData renderingData)
    {
        var controller = Object.FindAnyObjectByType<NeuralCapture>();
        if (controller != null)
        {
            m_ScriptablePass.controller = controller;
            m_ScriptablePass.depthConvertMaterial = depthConvertMaterial;
            m_ScriptablePass.ConfigureInput(ScriptableRenderPassInput.Motion | ScriptableRenderPassInput.Depth);
            renderer.EnqueuePass(m_ScriptablePass);
        }
    }
}
