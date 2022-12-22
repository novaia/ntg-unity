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

    [Header("Latent Vectors to Add")]
    [SerializeField] private bool BigMountainTopLeft;
    [SerializeField] private bool CentralValley;
    [SerializeField] private bool Lowlands;
    [SerializeField] private bool Highlands;
    [SerializeField] private bool DiagonalRidge;
    [SerializeField] private bool Highlands2;
    [SerializeField] private bool CentralValley2;
    [SerializeField] private bool BottomRightDecline;
    [SerializeField] private bool BottomRightDecline2;
    [SerializeField] private bool DivergingRidges;
    [SerializeField] private bool Highlands3;
    [SerializeField] private bool ValleyPass;
    [SerializeField] private bool CentralValley3;
    [SerializeField] private bool BottomLeftDecline;
    [SerializeField] private bool random;

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
        //Single[] heightmap = GenerateHeightmap(runtimeModel, AddTensors(InputTensorFromArray(latentVectors.CentralValley),
        //                                                                InputTensorFromArray(latentVectors.Highlands2)));
        Single[] heightmap = GenerateHeightmap(runtimeModel, CustomInputTensor());
        SetTerrainHeights(heightmap);
    }

    private void Update()
    {
        if(Input.GetKeyDown(KeyCode.Space))
        {
            Single[] heightmap = GenerateHeightmap(runtimeModel, CustomInputTensor());
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

    private Tensor AddTensors(Tensor a, Tensor b)
    {
        Tensor c = new Tensor(1, 100);
        for(int i = 0; i < 100; i++)
        {
            c[i] = a[i] + b[i];
        }
        return c;
    }

    private Tensor CustomInputTensor()
    {
        Tensor input = new Tensor(1, 100);

        if(BigMountainTopLeft)
        {
            input = AddTensors(input, InputTensorFromArray(latentVectors.BigMountainTopLeft));
        }
        if(CentralValley)
        {
            input = AddTensors(input, InputTensorFromArray(latentVectors.CentralValley));
        }
        if(Lowlands)
        {
            input = AddTensors(input, InputTensorFromArray(latentVectors.Lowlands));
        }
        if(Highlands)
        {
            input = AddTensors(input, InputTensorFromArray(latentVectors.Highlands));
        }
        if(DiagonalRidge)
        {
            input = AddTensors(input, InputTensorFromArray(latentVectors.DiagonalRidge));
        }
        if(Highlands2)
        {
            input = AddTensors(input, InputTensorFromArray(latentVectors.Highlands2));
        }
        if(CentralValley2)
        {
            input = AddTensors(input, InputTensorFromArray(latentVectors.CentralValley2));
        }
        if(BottomRightDecline)
        {
            input = AddTensors(input, InputTensorFromArray(latentVectors.BottomRightDecline));
        }
        if(BottomRightDecline2)
        {
            input = AddTensors(input, InputTensorFromArray(latentVectors.BottomRightDecline2));
        }
        if(DivergingRidges)
        {
            input = AddTensors(input, InputTensorFromArray(latentVectors.DivergingRidges));
        }        
        if(DivergingRidges)
        {
            input = AddTensors(input, InputTensorFromArray(latentVectors.DivergingRidges));
        }
        if(ValleyPass)
        {
            input = AddTensors(input, InputTensorFromArray(latentVectors.ValleyPass));
        }
        if(CentralValley3)
        {
            input = AddTensors(input, InputTensorFromArray(latentVectors.CentralValley3));
        }
        if(BottomLeftDecline)
        {
            input = AddTensors(input, InputTensorFromArray(latentVectors.BottomLeftDecline));
        }
        if(random)
        {
            input = AddTensors(input, RandomInputTensor());
        }

        return input;
    }
}
