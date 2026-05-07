Shader "Hidden/TemporalWarp"
{
    Properties
    {
        _MainTex ("Previous Styled Frame", 2D) = "black" {}
        _MotionTex ("Motion Vectors", 2D) = "black" {}
        _CurrentTex ("Current RGB Frame", 2D) = "black" {}
        _BlendFactor ("Temporal Blend", Range(0.05, 0.5)) = 0.15
    }
    SubShader
    {
        Cull Off ZWrite Off ZTest Always

        Pass
        {
            CGPROGRAM
            #pragma vertex vert
            #pragma fragment frag
            #include "UnityCG.cginc"

            struct appdata { float4 vertex : POSITION; float2 uv : TEXCOORD0; };
            struct v2f { float2 uv : TEXCOORD0; float4 vertex : SV_POSITION; };

            sampler2D _MainTex;
            float4 _MainTex_TexelSize;
            sampler2D _MotionTex;
            sampler2D _CurrentTex;
            float _BlendFactor;

            v2f vert (appdata v)
            {
                v2f o;
                o.vertex = UnityObjectToClipPos(v.vertex);
                o.uv = v.uv;
                return o;
            }

            fixed4 frag (v2f i) : SV_Target
            {
                // Extraer el vector de movimiento (asumiendo formato estándar de Unity)
                float2 motion = tex2D(_MotionTex, i.uv).rg;
                
                // Calcular UV del frame anterior
                float2 prevUV = i.uv - motion;
                
                // OOB (Out of Bounds) / Anti-Ghosting base
                if (prevUV.x < 0.0 || prevUV.x > 1.0 || prevUV.y < 0.0 || prevUV.y > 1.0)
                    return tex2D(_CurrentTex, i.uv); // Usar frame actual si OOB (no negro)
                
                fixed4 prevSample = tex2D(_MainTex, prevUV);
                fixed4 currentSample = tex2D(_CurrentTex, i.uv);

                // --- NEIGHBOR CLAMPING (FIX OLAS) ---
                // Muestrear vecinos del frame ACTUAL para definir el rango de color válido
                float2 tx = float2(_MainTex_TexelSize.x, 0);
                float2 ty = float2(0, _MainTex_TexelSize.y);
                fixed4 n0 = tex2D(_CurrentTex, i.uv + tx);
                fixed4 n1 = tex2D(_CurrentTex, i.uv - tx);
                fixed4 n2 = tex2D(_CurrentTex, i.uv + ty);
                fixed4 n3 = tex2D(_CurrentTex, i.uv - ty);

                fixed4 neighborMin = min(min(n0, n1), min(n2, n3));
                fixed4 neighborMax = max(max(n0, n1), max(n2, n3));
                neighborMin = min(neighborMin, currentSample);
                neighborMax = max(neighborMax, currentSample);

                // Clamp el frame anterior dentro del rango de vecinos actuales
                // Esto corta el ciclo de amplificación que genera "olas"
                fixed4 clampedPrev = clamp(prevSample, neighborMin, neighborMax);

                // Blend suave: más peso al frame anterior (estabilidad) vs actual (respuesta)
                // _BlendFactor = 0.15 → 85% previo, 15% actual
                return lerp(clampedPrev, currentSample, _BlendFactor);
            }
            ENDCG
        }
    }
}

