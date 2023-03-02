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
    private TensorMathHelper tensorMathHelper = new TensorMathHelper();

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

    private void BlendNeighbor(Terrain neighor, Tensor mirror, Tensor gradient, int xOffset, int yOffset)
    {
        Tensor localGradient = new Tensor(1, 256, 256, 1);
        for(int x = 0; x < 256; x++)
        {
            for(int y = 0; y < 256; y++)
            {
                localGradient[0, x, y, 0] = gradient[0, yOffset + y, xOffset + x, 0];
            }
        }
        Tensor scaled = tensorMathHelper.MultiplyTensors(localGradient, mirror);
        SetTerrainHeights(neighor, scaled.ToReadOnlyArray(), false);

        localGradient.Dispose();
        scaled.Dispose();
    }

    private void Blend(Terrain terrain)
    {
        float[,] heightmap = terrain.terrainData.GetHeights(0, 0, 256, 256);
        Tensor heightmapTensor = tensorMathHelper.TwoDimensionalArrayToTensor(heightmap);
        Tensor horizontalMirror = tensorMathHelper.MirrorTensor(heightmapTensor, false, true);
        Tensor verticalMirror = tensorMathHelper.MirrorTensor(heightmapTensor, true, false);
        Tensor bothMirror = tensorMathHelper.MirrorTensor(heightmapTensor, true, true);

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
        
        Terrain topLeftNeighbor = null;
        Terrain bottomLeftNeighbor = null;
        Terrain topRightNeighbor = null;
        Terrain bottomRightNeighbor = null;

        // Split gradient in 8 parts.
        Terrain leftNeighbor = terrain.leftNeighbor;
        if(leftNeighbor != null)
        {
            BlendNeighbor(leftNeighbor, horizontalMirror, gradient, 0, 256);

            topLeftNeighbor = leftNeighbor.topNeighbor;
            bottomLeftNeighbor = leftNeighbor.bottomNeighbor;
        }

        Terrain rightNeighbor = terrain.rightNeighbor;
        if(rightNeighbor != null)
        {
            BlendNeighbor(rightNeighbor, horizontalMirror, gradient, 512, 256);

            topRightNeighbor = rightNeighbor.topNeighbor;
            bottomRightNeighbor = rightNeighbor.bottomNeighbor;
        }

        Terrain topNeighbor = terrain.topNeighbor;
        if(topNeighbor != null)
        {
            BlendNeighbor(topNeighbor, verticalMirror, gradient, 256, 512);
        }

        Terrain bottomNeighbor = terrain.bottomNeighbor;
        if(bottomNeighbor != null)
        {
            BlendNeighbor(bottomNeighbor, verticalMirror, gradient, 256, 0);
        }
        
        if(topLeftNeighbor != null)
        {
            BlendNeighbor(topLeftNeighbor, bothMirror, gradient, 0, 512);
        }

        if(bottomLeftNeighbor != null)
        {
            BlendNeighbor(bottomLeftNeighbor, bothMirror, gradient, 0, 0);
        }

        if(topRightNeighbor != null)
        {
            BlendNeighbor(topRightNeighbor, bothMirror, gradient, 512, 512);
        }

        if(bottomRightNeighbor != null)
        {
            BlendNeighbor(bottomRightNeighbor, bothMirror, gradient, 512, 0);
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
