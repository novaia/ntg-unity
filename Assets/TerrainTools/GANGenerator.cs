using UnityEngine;
using UnityEditor;
using UnityEditor.TerrainTools;
using Unity.Barracuda;

class GANGenerator : TerrainPaintTool<GANGenerator>
{
    private int modelOutputWidth = 256;
    private int modelOutputHeight = 256;
    private NNModel modelAsset;
    private Model runtimeModel;
    private float heightMultiplier = 0.3f;
    private TensorMathHelper tensorMathHelper = new TensorMathHelper();

    public override string GetName()
    {
        return "NTG/" + "Gan Generator";
    }

    public override string GetDescription()
    {
        return "Generative adversarial network terrain generator.";
    }

    public override void OnInspectorGUI(Terrain terrain, IOnInspectorGUI editContext)
    {
        modelAsset = (NNModel)EditorGUILayout.ObjectField("Model Asset", modelAsset, typeof(NNModel), false);
        modelOutputWidth = EditorGUILayout.IntField("Model Output Width", modelOutputWidth);
        modelOutputHeight = EditorGUILayout.IntField("Model Output Height", modelOutputHeight);
        heightMultiplier = EditorGUILayout.FloatField("Height Multiplier", heightMultiplier);

        if(GUILayout.Button("Generate Terrain"))
        {
            float[] heightmap = GenerateHeightmap();
            SetTerrainHeights(terrain, heightmap);
        }
    }

    private float[] GenerateHeightmap()
    {
        if(runtimeModel == null)
        {
            if(modelAsset != null)
            {
                runtimeModel = ModelLoader.Load(modelAsset);
            }
            else 
            {
                Debug.LogError("Model asset is null.");
                return null;
            }
        }

        float[] heightmap = new float[modelOutputWidth * modelOutputHeight];
        using (var worker = WorkerFactory.CreateWorker(WorkerFactory.Type.ComputePrecompiled, runtimeModel))
        {
            Tensor input = tensorMathHelper.RandomNormalTensor(1, 1, 100, 1);
            worker.Execute(input);
            Tensor output = worker.PeekOutput();
            heightmap = output.ToReadOnlyArray();
            input.Dispose();
            output.Dispose();
        }

        for(int i = 0; i < 10; i++)
        {
            Debug.Log(heightmap[i]);
        }
        return heightmap;
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
