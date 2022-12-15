using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Unity.Barracuda;
using System;

public class TerrainGenerator : MonoBehaviour
{
    [SerializeField] private float heightMultiplier = 10.0f;
    [SerializeField] private GameObject terrainUnit;

    [SerializeField] private GameObject highTerrainUnit;
    [SerializeField] private GameObject middleTerrainUnit;
    [SerializeField] private GameObject lowTerrainUnit;

    [SerializeField] private float highThreshold;
    [SerializeField] private float lowThreshold;

    [SerializeField] private NNModel modelAsset;
    private Model runtimeModel;

    private const int modelOutputWidth = 256;
    private const int modelOutputHeight = 256;
    private const int modelOutputArea = modelOutputWidth * modelOutputHeight;

    private Single[] GenerateHeightmap(Model model)
    {
        // Reference: https://docs.unity3d.com/Packages/com.unity.barracuda@1.0/manual/Worker.html
        // Using ComputePrecompiled worker type for most efficient computation on GPU.
        var worker = WorkerFactory.CreateWorker(WorkerFactory.Type.ComputePrecompiled, model);

        // Model takes 1x100 noise vector.
        Tensor input = new Tensor(1, 100); 
        System.Random random = new System.Random();
        for(int i = 0; i < 100; i++)
        {
            input[0, i] = random.Next(0, 100) / 100f;
        }

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

    private void Start()
    {
        runtimeModel = ModelLoader.Load(modelAsset);
        Single[] heightmap = GenerateHeightmap(runtimeModel);

        for(int i = 0; i < modelOutputArea; i++)
        {
            int x = (int)(i % modelOutputWidth);
            int y = (int)Math.Floor((double)(i / modelOutputWidth));
            Vector3 terrainUnitPosition = new Vector3(x, heightmap[i] * heightMultiplier, y);
            if(terrainUnitPosition.y > highThreshold)
            {
                Instantiate(highTerrainUnit, terrainUnitPosition, Quaternion.identity);
            }
            else if(terrainUnitPosition.y < lowThreshold)
            {
                Instantiate(lowTerrainUnit, terrainUnitPosition, Quaternion.identity);
            }
            else
            {
                Instantiate(middleTerrainUnit, terrainUnitPosition, Quaternion.identity);
            }
        }
    }
}
