using UnityEngine;
using UnityEditor;
using UnityEditor.TerrainTools;
using Unity.Barracuda;

class BaseGenerator : TerrainPaintTool<BaseGenerator>
{
    protected int modelOutputWidth = 256;
    protected int modelOutputHeight = 256;
    protected NNModel modelAsset;
    protected Model runtimeModel;
    protected float heightMultiplier = 0.3f;

    // Name of the Terrain Tool. This appears in the tool UI.
    public override string GetName()
    {
        return "NTG/" + "Base Generator";
    }

    // Description for the Terrain Tool. This appears in the tool UI.
    public override string GetDescription()
    {
        return "Base generator for neural terrain generators.";
    }

    protected void DisplayUI()
    {
        modelOutputWidth = EditorGUILayout.IntField("Model Output Width", modelOutputWidth);
        modelOutputHeight = EditorGUILayout.IntField("Model Output Height", modelOutputHeight);
    }

    // Override this function to add UI elements to the inspector
    public override void OnInspectorGUI(Terrain terrain, IOnInspectorGUI editContext)
    {
        EditorGUI.BeginChangeCheck();
        DisplayUI();
        modelAsset = (NNModel)EditorGUILayout.ObjectField("Model Asset", modelAsset, typeof(NNModel), false);

        if(GUILayout.Button("Generate Terrain"))
        {
            object [] args = new object[] {};
            float [] heightmap = GenerateHeightmap(new WorkerExecuter(DefaultWorkerExecuter));
            SetTerrainHeights(terrain, heightmap);
        }

        if (EditorGUI.EndChangeCheck())
        {
            Setup();
        }
    }

    protected virtual void Setup()
    {
        if(modelAsset != null)
        {
            runtimeModel = ModelLoader.Load(modelAsset);
        }
    }

    protected delegate Tensor WorkerExecuter(IWorker worker, params object[] args);

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

    protected float[] GenerateHeightmap(WorkerExecuter workerExecuter, params object[] args)
    {
        if(runtimeModel == null)
        {
            Setup();
        }

        // Using ComputePrecompiled worker type for most efficient computation on GPU.
        // Reference: https://docs.unity3d.com/Packages/com.unity.barracuda@1.0/manual/Worker.html
        var worker = WorkerFactory.CreateWorker(WorkerFactory.Type.ComputePrecompiled, runtimeModel);

        Tensor output = workerExecuter(worker, args);
        float[] outputArray = output.ToReadOnlyArray();

        output.Dispose();
        worker.Dispose();

        return outputArray;
    }

    protected virtual Tensor DefaultWorkerExecuter(IWorker worker, params object[] args)
    {
        return new Tensor(1, modelOutputWidth, modelOutputHeight, 1);
    }
}