using UnityEngine;
using UnityEngine.Rendering;
using UnityEngine.Rendering.Universal;
using UnityEngine.Rendering.RenderGraphModule;

public class NeuralVisorFeature : ScriptableRendererFeature
{
    class NeuralVisorPass : ScriptableRenderPass
    {
        public NeuralSentisV5 neuralManager;
        public Material sharpenMaterial;
        public Material depthConvertMaterial; // FIX: LinearDepthCapture — igual que el dataset

        private class PassDataCapture { public TextureHandle src; public Material mat; }
        private class PassDataDrawing { public RTHandle aiRT; }

        public override void RecordRenderGraph(RenderGraph renderGraph, ContextContainer frameData)
        {
            if (neuralManager == null || !neuralManager.effectActive) return;

            // 1. Detección Inteligente de la Cámara Final
            UniversalCameraData cameraData = frameData.Get<UniversalCameraData>();
            var additionalData = cameraData.camera.GetUniversalAdditionalCameraData();
            
            // Solo las cámaras Base tienen stack. Las Overlay NO.
            bool hasOverlays = false;
            if (additionalData != null && additionalData.renderType == CameraRenderType.Base)
            {
                hasOverlays = additionalData.cameraStack != null && additionalData.cameraStack.Count > 0;
            }
            
            // Si la cámara es Base pero tiene Overlays, esperamos a que el arma se dibuje en el Overlay.
            bool isBaseWaitingForOverlay = (additionalData != null && additionalData.renderType == CameraRenderType.Base) && hasOverlays;
            
            if (isBaseWaitingForOverlay) return; 

            // 2. Garantía de Memoria
            if (neuralManager.currentRGB == null || !neuralManager.currentRGB.rt.IsCreated())
            {
                neuralManager.InitializeMemory();
                if (neuralManager.currentRGB == null) return;
            }

            if (!neuralManager.firstInferenceDone) return;

            UniversalResourceData resourceData = frameData.Get<UniversalResourceData>();
            TextureHandle cameraColor = resourceData.activeColorTexture;
            TextureHandle cameraDepth = resourceData.cameraDepthTexture;
            TextureHandle cameraMotion = resourceData.motionVectorColor;

            if (cameraData.cameraType != CameraType.Game || !cameraColor.IsValid()) return;

            // 3. Captura RGB
            using (var builder = renderGraph.AddRasterRenderPass<PassDataCapture>("Neural Capture RGB", out var passData))
            {
                passData.src = cameraColor;
                passData.mat = null;
                TextureHandle dst = renderGraph.ImportTexture(neuralManager.currentRGB);
                builder.UseTexture(cameraColor, AccessFlags.Read);
                builder.SetRenderAttachment(dst, 0, AccessFlags.Write);
                builder.AllowPassCulling(false);
                builder.SetRenderFunc((PassDataCapture data, RasterGraphContext context) => {
                    Blitter.BlitTexture(context.cmd, data.src, new Vector4(1, 1, 0, 0), 0f, false);
                });
            }

            // 4. Captura Depth — FIX: Usar LinearDepthCapture shader (igual que dataset training)
            // Sin este shader, el depth es raw hardware (Reverse Z) → casi negro → modelo recibe
            // datos completamente distintos a los que vio durante el entrenamiento.
            if (cameraDepth.IsValid())
            {
                using (var builder = renderGraph.AddRasterRenderPass<PassDataCapture>("Neural Capture Depth", out var passData))
                {
                    passData.src = cameraDepth;
                    passData.mat = depthConvertMaterial;
                    TextureHandle dst = renderGraph.ImportTexture(neuralManager.currentDepth);
                    builder.UseTexture(cameraDepth, AccessFlags.Read);
                    builder.SetRenderAttachment(dst, 0, AccessFlags.Write);
                    builder.AllowPassCulling(false);
                    builder.SetRenderFunc((PassDataCapture data, RasterGraphContext context) => {
                        if (data.mat != null)
                            Blitter.BlitTexture(context.cmd, data.src, new Vector4(1, 1, 0, 0), data.mat, 0);
                        else
                            Blitter.BlitTexture(context.cmd, data.src, new Vector4(1, 1, 0, 0), 0f, false);
                    });
                }
            }

            // 5. Dibujo Final (Blit de la IA a la pantalla)
            RTHandle outRT = neuralManager.GetOutputRTHandle();
            using (var builder = renderGraph.AddRasterRenderPass<PassDataDrawing>("Neural Display", out var passData))
            {
                passData.aiRT = outRT;
                TextureHandle aiHandle = renderGraph.ImportTexture(outRT);
                builder.UseTexture(aiHandle, AccessFlags.Read);
                builder.SetRenderAttachment(cameraColor, 0, AccessFlags.Write);
                builder.AllowPassCulling(false);
                builder.SetRenderFunc((PassDataDrawing data, RasterGraphContext context) => {
                    if (data.aiRT != null && data.aiRT.rt != null && data.aiRT.rt.IsCreated())
                        Blitter.BlitTexture(context.cmd, data.aiRT, new Vector4(1, 1, 0, 0), 0f, false);
                });
            }

            // 6. Sharpen Pass
            if (sharpenMaterial != null)
            {
                using (var builder = renderGraph.AddRasterRenderPass<PassDataCapture>("Neural Sharpen", out var passData))
                {
                    passData.src = cameraColor;
                    passData.mat = sharpenMaterial;
                    builder.UseTexture(cameraColor, AccessFlags.Read);
                    builder.SetRenderAttachment(cameraColor, 0, AccessFlags.Write);
                    builder.AllowPassCulling(false);
                    builder.SetRenderFunc((PassDataCapture data, RasterGraphContext context) => {
                        Blitter.BlitTexture(context.cmd, data.src, new Vector4(1, 1, 0, 0), data.mat, 0);
                    });
                }
            }
        }
    }

    public Material sharpenMaterial;
    public Material depthConvertMaterial; // Mismo shader que usa el dataset capture
    NeuralVisorPass m_ScriptablePass;

    public override void Create()
    {
        m_ScriptablePass = new NeuralVisorPass();
        m_ScriptablePass.renderPassEvent = RenderPassEvent.BeforeRenderingPostProcessing;

        // Auto-buscar el shader de conversión de depth si no está asignado
        if (depthConvertMaterial == null)
        {
            Shader s = Shader.Find("Hidden/Neural/LinearDepthCapture");
            if (s != null)
            {
                depthConvertMaterial = CoreUtils.CreateEngineMaterial(s);
                Debug.Log("<color=green>[Neural-Ink Visor] LinearDepthCapture shader encontrado y aplicado.</color>");
            }
            else
            {
                Debug.LogWarning("<color=orange>[Neural-Ink Visor] No se encontró 'Hidden/Neural/LinearDepthCapture'. " +
                                 "La profundidad en runtime NO coincidirá con el training.</color>");
            }
        }
    }

    public override void AddRenderPasses(ScriptableRenderer renderer, ref RenderingData renderingData)
    {
        var manager = Object.FindAnyObjectByType<NeuralSentisV5>();
        if (manager != null)
        {
            m_ScriptablePass.neuralManager        = manager;
            m_ScriptablePass.sharpenMaterial      = sharpenMaterial;
            m_ScriptablePass.depthConvertMaterial = depthConvertMaterial;
            renderer.EnqueuePass(m_ScriptablePass);
        }
    }
}
