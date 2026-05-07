Shader "Hidden/NeuralSharpen"
{
    Properties
    {
        _MainTex ("Texture", 2D) = "white" {}
        _Sharpness ("Sharpness", Range(0, 2)) = 1.0
    }
    SubShader
    {
        Tags { "RenderType"="Opaque" "RenderPipeline" = "UniversalPipeline"}
        LOD 100

        Pass
        {
            Name "SharpenPass"
            ZTest Always
            ZWrite Off
            Cull Off

            HLSLPROGRAM
            #pragma vertex vert
            #pragma fragment frag
            #include "Packages/com.unity.render-pipelines.universal/ShaderLibrary/Core.hlsl"

            struct Attributes
            {
                float4 positionOS : POSITION;
                float2 uv : TEXCOORD0;
            };

            struct Varyings
            {
                float4 positionHCS : SV_POSITION;
                float2 uv : TEXCOORD0;
            };

            sampler2D _MainTex;
            float4 _MainTex_ST;
            float4 _MainTex_TexelSize;
            float _Sharpness;

            Varyings vert (Attributes input)
            {
                Varyings output;
                output.positionHCS = TransformObjectToHClip(input.positionOS.xyz);
                output.uv = TRANSFORM_TEX(input.uv, _MainTex);
                return output;
            }

            float4 frag (Varyings input) : SV_Target
            {
                float2 uv = input.uv;
                float2 texel = _MainTex_TexelSize.xy;

                // Muestreo de píxeles vecinos (Filtro de Nitidez tipo Laplace)
                float4 center = tex2D(_MainTex, uv);
                float4 left   = tex2D(_MainTex, uv + float2(-texel.x, 0));
                float4 right  = tex2D(_MainTex, uv + float2(texel.x, 0));
                float4 up     = tex2D(_MainTex, uv + float2(0, texel.y));
                float4 down   = tex2D(_MainTex, uv + float2(0, -texel.y));

                // Algoritmo de nitidez: 5*Centro - (Arriba + Abajo + Izquierda + Derecha)
                float4 sharpened = center + (center * 4 - (left + right + up + down)) * _Sharpness;

                return float4(sharpened.rgb, 1.0);
            }
            ENDHLSL
        }
    }
}
