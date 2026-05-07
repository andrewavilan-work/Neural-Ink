Shader "NeuralInk/MotionVectorExport"
{
    SubShader
    {
        Tags { "RenderPipeline" = "UniversalPipeline" }
        Pass
        {
            HLSLPROGRAM
            #pragma vertex vert
            #pragma fragment frag
            #include "Packages/com.unity.render-pipelines.universal/ShaderLibrary/Core.hlsl"

            struct Attributes {
                float4 positionOS : POSITION;
                float2 uv : TEXCOORD0;
            };

            struct Varyings {
                float4 positionCS : SV_POSITION;
                float2 uv : TEXCOORD0;
            };

            // URP automáticamente llena esta textura si el depthTextureMode tiene MotionVectors
            TEXTURE2D(_MotionVectorTexture);
            SAMPLER(sampler_MotionVectorTexture);

            Varyings vert(Attributes input)
            {
                Varyings output;
                output.positionCS = TransformObjectToHClip(input.positionOS.xyz);
                output.uv = input.uv;
                return output;
            }

            float4 frag(Varyings input) : SV_Target
            {
                // Leer vectores de movimiento (XY)
                float2 mv = SAMPLE_TEXTURE2D(_MotionVectorTexture, sampler_MotionVectorTexture, input.uv).xy;
                
                // Mapear el rango [-1, 1] a [0, 1] para poder guardarlo en un PNG estándar
                // R = Movimiento X, G = Movimiento Y, B = 0.5 (neutro)
                return float4(mv * 0.5 + 0.5, 0.5, 1.0);
            }
            ENDHLSL
        }
    }
}
