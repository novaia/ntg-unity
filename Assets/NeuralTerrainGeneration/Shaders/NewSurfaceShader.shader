Shader "TerrainTool/CustomTerrainTool"
{
    Properties { _MainTex ("Texture", any) = "" {} }

    SubShader
    {
        ZTest Always Cull Off ZWrite Off

        HLSLINCLUDE

        #include "UnityCG.cginc"
        #include "Packages/com.unity.terrain-tools/Shaders/TerrainTools.hlsl"

        sampler2D _MainTex;
        float4 _MainTex_TexelSize;      // 1/width, 1/height, width, height

        sampler2D _BrushTex;

        float4 _BrushParams;
        #define BRUSH_STRENGTH      (_BrushParams[0])
        #define BRUSH_TARGETHEIGHT  (_BrushParams[1])
        #define kMaxHeight          (32766.0f/65535.0f)

        struct appdata_t
        {
            float4 vertex : POSITION;
            float2 pcUV : TEXCOORD0;
        };

        struct v2f
        {
            float4 vertex : SV_POSITION;
            float2 pcUV : TEXCOORD0;
        };

        v2f vert(appdata_t v)
        {
            v2f o;
            o.vertex = UnityObjectToClipPos(v.vertex);
            o.pcUV = v.pcUV;
            return o;
        }

        ENDHLSL

        Pass
        {
            Name "CustomTerrainTool"

            HLSLPROGRAM

            #pragma vertex vert
            #pragma fragment frag

            float4 frag(v2f i) : SV_Target
            {
                float2 brushUV = PaintContextUVToBrushUV(i.pcUV);

                // out of bounds multiplier
                float oob = all(saturate(brushUV) == brushUV) ? 1.0f : 0.0f;

                // Sample the MainTex, which should be a region of the source Heightmap texture, to get the current height value at the provided UV
                // UnpackHeightmap is necessary here because it unpacks the height value from R and G channels if the current platform/graphics device does not support R16_UNorm texture formats. If R16_UNorm formats are supported, UnpackHeightmap just reads from the R channel.
                float height = UnpackHeightmap(tex2D(_MainTex, i.pcUV));
                // Calculate the influence from the brush mask at this fragment
                float brushShape = oob * UnpackHeightmap(tex2D(_BrushTex, brushUV));
                // Calculate the new height value
                float brushHeight = 0.1f * brushShape + 0.25f;
                //height = (height > 0.28f) ? height : 0.28f;
                height = (height > brushHeight) ? height : brushHeight;
                
                //height = height + 0.1f * brushShape;

                //height = 0.3f;

                // Store the new height into the destination RenderTexture. Clamp between 0.0f and 0.5f because the Heightmap itself is signed but is treated as an unsigned texture when rendering the Terrain
                // PackHeightmap is necessary here because it packs the height value into R and G channels if the current platform/graphics device does not support R16_UNorm texture formats. If R16_UNorm formats are supported, PackHeightmap just writes to the R channel.
                return PackHeightmap(clamp(height, 0, kMaxHeight));
            }

            ENDHLSL
        }
    }
}