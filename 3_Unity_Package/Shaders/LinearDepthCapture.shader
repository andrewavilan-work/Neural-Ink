Shader "Hidden/Neural/LinearDepthCapture"
{
    SubShader
    {
        Tags { "RenderType"="Opaque" "RenderPipeline" = "UniversalPipeline"}
        LOD 100
        ZTest Always ZWrite Off Cull Off

        Pass
        {
            Name "DepthToLinear"

            HLSLPROGRAM
            #include "Packages/com.unity.render-pipelines.universal/ShaderLibrary/Core.hlsl"
            #include "Packages/com.unity.render-pipelines.core/Runtime/Utilities/Blit.hlsl"

            #pragma vertex Vert
            #pragma fragment frag

            float frag (Varyings input) : SV_Target
            {
                UNITY_SETUP_STEREO_EYE_INDEX_POST_VERTEX(input);
                
                // Leer el depth raw
                float depth = SAMPLE_TEXTURE2D_X(_BlitTexture, sampler_LinearClamp, input.texcoord).r;
                
                // 1. Convertir a Profundidad Lineal de Ojo (en metros reales)
                float eyeDepth = LinearEyeDepth(depth, _ZBufferParams);

                // 2. Aplicar escala logarítmica para visibilidad comercial
                // Esto comprime el rango de 0-1000 metros en algo que el ojo (y la IA)
                // pueden ver con mucho más contraste.
                // Usamos un factor de 2.0 para resaltar detalles de media distancia.
                float logDepth = log2(1.0 + eyeDepth) * 0.1; 

                // Asegurar que nos mantenemos en el rango 0-1
                return saturate(logDepth);
            }
            ENDHLSL
        }
    }
}
