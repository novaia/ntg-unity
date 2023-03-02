using UnityEngine;
using UnityEditor;
using UnityEditor.TerrainTools;
using Unity.Barracuda;

class BlendWithNeighbors : TerrainPaintTool<BlendWithNeighbors>
{
    private float radius1 = 128.0f;
    private float radius2 = 256.0f;
    private int modelOutputHeight = 256;
    private int modelOutputWidth = 256;
    private float heightMultiplier = 0.3f;
    private float bValue = 1.0f;

    public override string GetName()
    {
        return "NTG/" + "Blend With Neighbors";
    }

    public override string GetDescription()
    {
        return "Blends the selected terrain with its eight neighors.";
    }

    public override void OnInspectorGUI(Terrain terrain, IOnInspectorGUI editContext)
    {
        bValue = EditorGUILayout.FloatField("B Value", bValue);
        if(GUILayout.Button("Blend With Neighbors"))
        {
            Blend(terrain);
        }
    }

    private void Blend(Terrain terrain)
    {
        TensorMathHelper tensorMathHelper = new TensorMathHelper();

        float[,] heightmap = terrain.terrainData.GetHeights(0, 0, 256, 256);
        Tensor heightmapTensor = tensorMathHelper.TwoDimensionalArrayToTensor(heightmap);
        Tensor horizontalMirror = tensorMathHelper.MirrorTensor(heightmapTensor, false, true);
        Tensor verticalMirror = tensorMathHelper.MirrorTensor(heightmapTensor, true, false);

        Tensor gradient = new Tensor(1, 256 * 3, 256 * 3, 1);
        Vector2 center = new Vector2(radius1 + radius2, radius1 + radius2);
        for(int x = 0; x < 256 * 3; x++)
        {
            for(int y = 0; y < 256 * 3; y++)
            {
                float distance = Vector2.Distance(new Vector2(x, y), center);
                if(distance < radius1)
                {
                    gradient[0, x, y, 0] = 1.0f;
                }
                else
                {
                    float gradientValue = (-1.0f / 128.0f) * distance + bValue;
                    if(gradientValue > 1.0f)
                    {
                        gradient[0, x, y, 0] = 1.0f;
                    }
                    else
                    {
                        gradient[0, x, y, 0] = gradientValue;
                    }
                }
            }
        }

        Tensor gradientTest = tensorMathHelper.GradientTensor(0.0f, 0.0f, 1.0f, 0.0f, 256, 256);

        // Split gradient in 8 parts.
        Terrain leftNeighbor = terrain.leftNeighbor;
        if(leftNeighbor != null)
        {
            // Left.
            Tensor leftGradient = new Tensor(1, 256, 256, 1);
            for(int x = 0; x < 256; x++)
            {
                for(int y = 0; y < 256; y++)
                {
                    leftGradient[0, x, y, 0] = gradient[0, 256 + y, x, 0];
                }
            }
            Tensor leftScaled = tensorMathHelper.MultiplyTensors(leftGradient, horizontalMirror);
            SetTerrainHeights(leftNeighbor, leftScaled.ToReadOnlyArray(), false);
        }

        Terrain rightNeighbor = terrain.rightNeighbor;
        if(rightNeighbor != null)
        {
            // Right.
            Tensor rightGradient = new Tensor(1, 256, 256, 1);
            for(int x = 0; x < 256; x++)
            {
                for(int y = 0; y < 256; y++)
                {
                    rightGradient[0, x, y, 0] = gradient[0, 256 + y, 256 * 2 + x, 0];
                }
            }
            Tensor rightScaled = tensorMathHelper.MultiplyTensors(rightGradient, horizontalMirror);
            SetTerrainHeights(rightNeighbor, rightScaled.ToReadOnlyArray(), false);
        }

        Terrain topNeighbor = terrain.topNeighbor;
        if(topNeighbor != null)
        {
            // Top.
            Tensor topGradient = new Tensor(1, 256, 256, 1);
            for(int x = 0; x < 256; x++)
            {
                for(int y = 0; y < 256; y++)
                {
                    topGradient[0, x, y, 0] = gradient[0, 256 * 2 + y, 256 + x, 0];
                }
            }
            Tensor topScaled = tensorMathHelper.MultiplyTensors(topGradient, verticalMirror);
            SetTerrainHeights(topNeighbor, topScaled.ToReadOnlyArray(), false);
        }

        Terrain bottomNeighbor = terrain.bottomNeighbor;
        if(bottomNeighbor != null)
        {
            // Bottom.
            Tensor bottomGradient = new Tensor(1, 256, 256, 1);
            for(int x = 0; x < 256; x++)
            {
                for(int y = 0; y < 256; y++)
                {
                    bottomGradient[0, x, y, 0] = gradient[0, y, 256 + x, 0];
                }
            }
            Tensor bottomScaled = tensorMathHelper.MultiplyTensors(bottomGradient, verticalMirror);
            SetTerrainHeights(bottomNeighbor, bottomScaled.ToReadOnlyArray(), false);
        }
    }

    public void SetTerrainHeights(Terrain terrain, float[] heightmap, bool scale = true)
    {
        terrain.terrainData.heightmapResolution = modelOutputWidth;

        float scaleCoefficient = 1;
        if(scale)
        {
            float maxValue = heightmap[0];
            for(int i = 0; i < heightmap.Length; i++)
            {
                if(heightmap[i] > maxValue)
                {
                    maxValue = heightmap[i];
                }
            }
            scaleCoefficient = (1 / maxValue) * heightMultiplier;
        }

        float[,] newHeightmap = new float[modelOutputWidth+1, modelOutputHeight+1];
        for(int x = 0; x < modelOutputWidth; x++)
        {
            for(int y = 0; y < modelOutputHeight; y++)
            {
                newHeightmap[x, y] = heightmap[x + y * modelOutputWidth] * scaleCoefficient;
            }
        }

        for(int i = 0; i < modelOutputWidth+1; i++)
        {
            newHeightmap[i, modelOutputHeight] = newHeightmap[i, modelOutputHeight-1];
        }
        for(int i = 0; i < modelOutputHeight+1; i++)
        {
            newHeightmap[modelOutputWidth, i] = newHeightmap[modelOutputWidth-1, i];
        }

        terrain.terrainData.SetHeights(0, 0, newHeightmap);
    }
}
