using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Unity.Barracuda;
using System;

public class TerrainGeneratorForUnityTerrain : MonoBehaviour
{
    [SerializeField] private Terrain terrain;

    [SerializeField] private float heightMultiplier = 10.0f;

    [SerializeField] private float highThreshold;
    [SerializeField] private float lowThreshold;

    [SerializeField] private NNModel modelAsset;
    private Model runtimeModel;

    private const int modelOutputWidth = 256;
    private const int modelOutputHeight = 256;
    private const int modelOutputArea = modelOutputWidth * modelOutputHeight;

    [SerializeField] private LatentVectors latentVectors;

    private Single[] GenerateHeightmap(Model model, Tensor input)
    {
        // Reference: https://docs.unity3d.com/Packages/com.unity.barracuda@1.0/manual/Worker.html
        // Using ComputePrecompiled worker type for most efficient computation on GPU.
        var worker = WorkerFactory.CreateWorker(WorkerFactory.Type.ComputePrecompiled, model);

        // Execute model.
        worker.Execute(input);

        // Fetch output.
        Tensor output = worker.CopyOutput();
        Single[] map = output.ToReadOnlyArray();

        // Dispose of tensor data.
        worker.Dispose();
        input.Dispose();

        return map;
    }

    private void SetTerrainHeights(Single[] heightmap)
    {
        float[,] newHeightmap = new float[modelOutputWidth, modelOutputHeight];
        for(int i = 0; i < modelOutputArea; i++)
        {
            int x = (int)(i % modelOutputWidth);
            int y = (int)Math.Floor((double)(i / modelOutputWidth));
            newHeightmap[x, y] = (float)heightmap[i] * heightMultiplier;
        }

        terrain.terrainData.SetHeights(0, 0, newHeightmap);
    }

    private void Start()
    {
        terrain.terrainData.heightmapResolution = 256;
        runtimeModel = ModelLoader.Load(modelAsset);
        Single[] heightmap = GenerateHeightmap(runtimeModel, InputTensorFromArray(latentVectors.BottomLeftDecline));
        SetTerrainHeights(heightmap);
    }

    private void Update()
    {
        if(Input.GetKeyDown(KeyCode.Space))
        {
            Single[] heightmap = GenerateHeightmap(runtimeModel, RandomInputTensor());
            SetTerrainHeights(heightmap); 
        }
    }

    private Tensor RandomInputTensor()
    {
        Tensor input = new Tensor(1, 100);        
        System.Random random = new System.Random();
        for(int i = 0; i < 100; i++)
        {
            input[0, i] = random.Next(0, 100) / 100f;
        }
        return input;  
    }

    private Tensor InputTensorFromArray(float[] inputArray)
    {
        Tensor input = new Tensor(1, 100);
        for(int i = 0; i < inputArray.Length; i++)
        {
            input[i] = inputArray[i];
        }
        return input;
    }
}
