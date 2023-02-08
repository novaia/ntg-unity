using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System;
using Unity.Barracuda;

public class BaseTerrainGenerator : MonoBehaviour
{
    [SerializeField] protected NNModel modelAsset;
    protected Model runtimeModel;

    protected TensorMathHelper tensorMathHelper = new TensorMathHelper();

    [SerializeField] protected int modelOutputWidth = 256;
    [SerializeField] protected int modelOutputHeight = 256;
    protected int modelOutputArea;
    protected int channels = 1;

    public Terrain terrain;
    [SerializeField] protected float heightMultiplier = 0.3f;

    protected delegate Tensor WorkerExecuter(IWorker worker, params object[] args);

    public virtual void Setup()
    {
        terrain = GetComponent<Terrain>();
        modelOutputArea = modelOutputWidth * modelOutputHeight;
        if(modelAsset != null)
        {
            runtimeModel = ModelLoader.Load(modelAsset);
        }
    }

    public void SetTerrainHeights(Single[] heightmap, bool scale = true)
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

    protected Single[] GenerateHeightmap(Model model, WorkerExecuter workerExecuter, params object[] args)
    {
        // Using ComputePrecompiled worker type for most efficient computation on GPU.
        // Reference: https://docs.unity3d.com/Packages/com.unity.barracuda@1.0/manual/Worker.html
        var worker = WorkerFactory.CreateWorker(WorkerFactory.Type.ComputePrecompiled, model);

        Tensor output = workerExecuter(worker, args);
        Single[] outputArray = output.ToReadOnlyArray();

        output.Dispose();
        worker.Dispose();

        return outputArray;
    }
    
    protected virtual Tensor DefaultWorkerExecuter(IWorker worker, params object[] args)
    {
        return new Tensor(1, modelOutputWidth, modelOutputHeight, 1);
    }
}
